use anyhow::{Context, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::asr::AsrProvider;
use crate::audio::AudioCapture;
use crate::common::types::{HotkeyEvent, RecordingState};
use crate::hotkey::{
    check_accessibility_permission, request_accessibility_permission, HotkeyListener,
};
use crate::text_injection::TextInjector;
use crate::tray::{TrayHandle, TrayIconState};

/// Daemon event types
#[derive(Debug)]
pub enum DaemonEvent {
    Hotkey(HotkeyEvent),
    TranscriptionComplete(String),
    /// Hotkey listener thread has exited (crashed or channel disconnected), daemon should stop
    HotkeyListenerDead,
}

pub struct Daemon {
    state: Arc<Mutex<RecordingState>>,
    /// Ignore new hotkeys while transcription/paste is in progress, avoiding race conditions
    is_processing: AtomicBool,
    /// Number of hotkey events received (for logging: Nth keypress)
    hotkey_recv_count: std::sync::atomic::AtomicU64,
    audio_capture: AudioCapture,
    provider: Arc<dyn AsrProvider>,
    text_injector: TextInjector,
    /// Current recording stream (Some = recording, drop to stop)
    active_stream: Mutex<Option<cpal::Stream>>,
    /// Current recording session buffer (new Arc per recording, prevents stale stream callbacks from polluting new session)
    recording_buffer: Mutex<Arc<Mutex<Vec<f32>>>>,
    /// Tray handle (Send+Sync, state updates sent back to main thread)
    tray: Option<Arc<TrayHandle>>,
    /// Trigger mode: "toggle" or "hold"
    trigger_mode: String,
}

impl Daemon {
    pub fn new(
        provider: Arc<dyn AsrProvider>,
        tray: Option<Arc<TrayHandle>>,
    ) -> Result<Self> {
        let audio_capture = AudioCapture::new().context("Failed to initialize audio capture")?;
        let text_injector = TextInjector::new();
        let config = crate::common::config::Config::load().unwrap_or_default();

        Ok(Self {
            state: Arc::new(Mutex::new(RecordingState::default())),
            is_processing: AtomicBool::new(false),
            hotkey_recv_count: std::sync::atomic::AtomicU64::new(0),
            audio_capture,
            provider,
            text_injector,
            active_stream: Mutex::new(None),
            recording_buffer: Mutex::new(Arc::new(Mutex::new(Vec::new()))),
            tray,
            trigger_mode: config.trigger_mode,
        })
    }

    pub async fn run(self) -> Result<()> {
        let current_exe = std::env::current_exe()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|e| format!("<unavailable: {e}>"));
        let accessibility_ok = check_accessibility_permission();
        let input_monitoring_ok = crate::hotkey::check_input_monitoring_permission();

        info!(
            "Permission diagnostics: current_exe={} accessibility_ok={} input_monitoring_ok={}",
            current_exe, accessibility_ok, input_monitoring_ok
        );
        let microphone_ok = crate::hotkey::check_microphone_permission();
        println!("🔎 Permission diagnostics");
        println!("   Executable: {}", current_exe);
        println!("   Accessibility: {}", accessibility_ok);
        println!("   Input Monitoring: {}", input_monitoring_ok);
        println!("   Microphone: {}", microphone_ok);

        // Request missing permissions (triggers system dialog only if not yet granted)
        if !accessibility_ok {
            println!();
            println!("⚠️  Accessibility permission not granted — requesting...");
            request_accessibility_permission();
        }
        if !input_monitoring_ok {
            println!();
            println!("⚠️  Input Monitoring permission not granted — requesting...");
            crate::hotkey::request_input_monitoring_permission();
        }
        if !microphone_ok {
            println!();
            println!("⚠️  Microphone permission not yet granted.");
            crate::hotkey::request_microphone_permission();
        }

        if !accessibility_ok || !input_monitoring_ok {
            println!();
            println!("⏳ Waiting for permissions... Grant them in System Settings, then the app will retry.");
            println!("   (If the hotkey doesn't work after granting, restart the app)");
            // Don't bail — let CGEventTap creation fail with a clear error instead
        }

        // ── Provider status check ──────────────────────────────────────────
        if let Err(e) = self.provider.check_status() {
            anyhow::bail!("Provider not ready: {}", e);
        }

        // ── Provider warmup (eliminate first inference JIT overhead) ──────────────────────
        {
            let warmup_start = std::time::Instant::now();
            self.provider.warmup().await?;
            let warmup_ms = warmup_start.elapsed().as_millis();
            info!("Provider warmup: {}ms", warmup_ms);
        }

        // ── Audio device info ──────────────────────────────────────────────
        let audio_info = self.audio_capture.get_info();

        // ── Start hotkey listener ─────────────────────────────────────────────
        let config = crate::common::config::Config::load().unwrap_or_default();
        let (hotkey_tx, hotkey_rx) = std::sync::mpsc::channel();
        let listener = HotkeyListener::new(hotkey_tx, config.hotkey.clone());
        listener.start().context("Failed to start hotkey listener")?;

        // ── Bridge sync mpsc to tokio mpsc ─────────────────────────────
        let (event_tx, mut event_rx) = mpsc::channel::<DaemonEvent>(32);
        let event_tx_clone = event_tx.clone();
        tokio::task::spawn_blocking(move || loop {
            match hotkey_rx.recv() {
                Ok(ev) => {
                    if event_tx_clone
                        .blocking_send(DaemonEvent::Hotkey(ev))
                        .is_err()
                    {
                        break;
                    }
                }
                Err(_) => {
                    // Hotkey listener thread has exited (channel sender dropped), notify main loop
                    let _ = event_tx_clone.blocking_send(DaemonEvent::HotkeyListenerDead);
                    break;
                }
            }
        });

        // ── Ready notification ──────────────────────────────────────────────────
        println!();
        println!("✅ Open Flow is ready");
        println!(
            "   Audio device: {}Hz / {} channels",
            audio_info.sample_rate, audio_info.channels
        );
        println!("   Provider: {}", self.provider.name());
        println!();
        println!("🎙️  Press hotkey to start recording, press again to stop and transcribe");
        println!("   Tray icon shows status (gray=idle, red=recording, yellow=transcribing)");
        println!();

        // ── Main event loop ────────────────────────────────────────────────
        loop {
            tokio::select! {
                Some(event) = event_rx.recv() => {
                    match event {
                        DaemonEvent::Hotkey(ev) => {
                            self.handle_hotkey(ev, &event_tx).await;
                        }
                        DaemonEvent::TranscriptionComplete(text) => {
                            self.is_processing.store(false, Ordering::SeqCst);
                            self.set_tray(TrayIconState::Idle);
                            println!("📝 Transcription complete: {}", text);
                            // Short delay before typing to let the user release the hotkey
                            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                            info!("[Hotkey] Typing started");
                            if let Err(e) = self.text_injector.inject(&text).await {
                                eprintln!("⚠️  Text injection failed: {e}");
                            }
                            info!("[Hotkey] Paste finished");
                        }
                        DaemonEvent::HotkeyListenerDead => {
                            eprintln!("❌ Hotkey listener thread has exited, daemon stopping. Please run open-flow start to restart.");
                            break;
                        }
                    }
                }
                        // daemon checks tray exit flag every 200ms
                _ = tokio::time::sleep(std::time::Duration::from_millis(200)) => {
                    if self.tray.as_ref().map_or(false, |t| t.exit_requested()) {
                        println!("👋 Tray exit signal received, daemon stopping...");
                        break;
                    }
                }
            }
        }

        Ok(())
    }

    fn set_tray(&self, state: TrayIconState) {
        if let Some(ref t) = self.tray {
            t.set_state(state);
        }
    }

    async fn handle_hotkey(&self, event: HotkeyEvent, tx: &mpsc::Sender<DaemonEvent>) {
        let n = self.hotkey_recv_count.fetch_add(1, Ordering::SeqCst) + 1;
        let is_processing = self.is_processing.load(Ordering::SeqCst);
        let is_recording = self.state.lock().unwrap().is_recording;
        info!(
            "[Hotkey] #{} event={:?} is_recording={} is_processing={} mode={}",
            n, event, is_recording, is_processing, self.trigger_mode
        );

        match event {
            HotkeyEvent::Pressed => {
                if is_processing {
                    info!("[Hotkey] #{} -> ignored (processing)", n);
                    return;
                }
                if self.trigger_mode == "hold" {
                    // Hold mode: Pressed always starts recording
                    if !is_recording {
                        info!("[Hotkey] #{} -> start recording (hold mode)", n);
                        if let Err(e) = self.start_recording() {
                            eprintln!("⚠️  Recording start failed: {e}");
                        } else {
                            self.set_tray(TrayIconState::Recording);
                        }
                    }
                } else {
                    // Toggle mode: Pressed toggles state
                    if is_recording {
                        info!("[Hotkey] #{} -> stop and transcribe (toggle mode)", n);
                        self.set_tray(TrayIconState::Transcribing);
                        if let Err(e) = self.stop_and_transcribe(tx).await {
                            self.set_tray(TrayIconState::Idle);
                            eprintln!("⚠️  Transcription failed: {e}");
                        }
                    } else {
                        info!("[Hotkey] #{} -> start recording (toggle mode)", n);
                        if let Err(e) = self.start_recording() {
                            eprintln!("⚠️  Recording start failed: {e}");
                        } else {
                            self.set_tray(TrayIconState::Recording);
                        }
                    }
                }
            }
            HotkeyEvent::Released => {
                if self.trigger_mode == "hold" && is_recording {
                    // Hold mode: Released always stops, even if processing
                    info!("[Hotkey] #{} -> stop and transcribe (hold release)", n);
                    self.set_tray(TrayIconState::Transcribing);
                    if let Err(e) = self.stop_and_transcribe(tx).await {
                        self.set_tray(TrayIconState::Idle);
                        eprintln!("⚠️  Transcription failed: {e}");
                    }
                }
                // Toggle mode: Released is ignored
            }
        }
    }

    fn start_recording(&self) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        if state.is_recording {
            return Ok(());
        }

        // Create a fresh Arc for each recording, stale callbacks from old stream only write to old Arc, not affecting this session
        let session_buf: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
        let stream = self
            .audio_capture
            .build_live_stream(session_buf.clone())
            .context("Failed to create recording stream")?;
        *self.recording_buffer.lock().unwrap() = session_buf;
        *self.active_stream.lock().unwrap() = Some(stream);

        state.is_recording = true;
        state.start_time = Some(std::time::Instant::now());

        info!("[Hotkey] Recording started");
        println!("🔴 Recording... press hotkey again to stop");
        Ok(())
    }

    async fn stop_and_transcribe(&self, tx: &mpsc::Sender<DaemonEvent>) -> Result<()> {
        self.is_processing.store(true, Ordering::SeqCst);

        let duration = {
            let mut state = self.state.lock().unwrap();
            if !state.is_recording {
                return Ok(());
            }
            state.is_recording = false;
            state
                .start_time
                .map(|t| t.elapsed())
                .unwrap_or_default()
        };

        // Take this session's Arc first, then drop the stream
        // Even if old stream has stale callbacks that continue writing, they write to the old Arc, not affecting the next session
        let session_buf = self.recording_buffer.lock().unwrap().clone();
        drop(self.active_stream.lock().unwrap().take());

        let buffer: Vec<f32> = session_buf.lock().unwrap().clone();
        info!(
            "[Hotkey] Recording stopped, starting transcription (duration {:.1}s, {} samples)",
            duration.as_secs_f32(),
            buffer.len()
        );
        println!(
            "⏹️  Recording stopped ({:.1}s / {} samples), transcribing...",
            duration.as_secs_f32(),
            buffer.len()
        );

        if buffer.is_empty() {
            self.is_processing.store(false, Ordering::SeqCst);
            eprintln!("⚠️  Recording is empty, please check microphone permission (System Settings > Privacy & Security > Microphone)");
            return Ok(());
        }

        // Amplitude diagnostics: if volume is too low, microphone permission may be missing or muted
        let max_amp = buffer.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let rms = (buffer.iter().map(|x| x * x).sum::<f32>() / buffer.len() as f32).sqrt();
        info!("[Audio] max_amp={:.6} rms={:.6}", max_amp, rms);
        if max_amp < 0.001 {
            self.is_processing.store(false, Ordering::SeqCst);
            eprintln!("⚠️  Recording amplitude extremely low (max={:.6}), microphone may not be authorized.", max_amp);
            eprintln!("   Please go to: System Settings > Privacy & Security > Microphone, add Open Flow.app to the list and enable it.");
            eprintln!("   Then fully quit and reopen Open Flow.");
            return Ok(());
        }

        // Transcribe via AsrProvider trait (local or cloud)
        let sample_rate = self.audio_capture.get_info().sample_rate;
        let provider = self.provider.clone();

        let result = match tokio::time::timeout(
            std::time::Duration::from_secs(30),
            provider.transcribe(&buffer, sample_rate),
        )
        .await
        {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                self.is_processing.store(false, Ordering::SeqCst);
                return Err(e);
            }
            Err(_elapsed) => {
                self.is_processing.store(false, Ordering::SeqCst);
                eprintln!("⚠️  Transcription timed out (>30s), abandoned. Please check model or restart daemon.");
                return Ok(());
            }
        };

        tx.send(DaemonEvent::TranscriptionComplete(result.text))
            .await
            .ok();

        Ok(())
    }
}

pub async fn run_daemon(
    provider: Arc<dyn AsrProvider>,
    tray: Option<Arc<TrayHandle>>,
) -> Result<()> {
    let daemon = Daemon::new(provider, tray)?;
    daemon.run().await
}

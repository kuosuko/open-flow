# Open Flow Enhancements Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Groq API provider, configurable hotkeys (Fn key), hold/toggle trigger modes, floating indicator overlay, and native settings window to open-flow.

**Architecture:** Introduce an `AsrProvider` trait to abstract local vs cloud transcription. Refactor the hotkey system to support configurable keys and press/release events. Add native macOS NSPanel overlay and NSWindow settings UI using existing `cocoa`/`objc` crates. All new macOS-specific features follow the existing stub pattern for cross-platform compilation.

**Tech Stack:** Rust 2021, tokio, cocoa/objc (macOS native UI), reqwest (Groq API), cpal (audio), ONNX Runtime (local ASR)

**Spec:** `docs/superpowers/specs/2026-03-14-open-flow-enhancements-design.md`

---

## Chunk 1: Provider Abstraction + Groq Backend

### Task 1: Expand Config struct

**Files:**
- Modify: `src/common/config.rs`

- [ ] **Step 1: Add new config fields**

```rust
// In src/common/config.rs, update the Config struct:

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub model_path: Option<PathBuf>,
    /// "local" or "groq"
    #[serde(default = "default_provider")]
    pub provider: String,
    /// Groq API key (env var GROQ_API_KEY takes precedence)
    #[serde(default)]
    pub groq_api_key: String,
    /// Groq Whisper model name
    #[serde(default = "default_groq_model")]
    pub groq_model: String,
    /// Optional language hint for Groq (e.g. "en", "zh"). Empty = auto-detect.
    #[serde(default)]
    pub groq_language: String,
    /// Hotkey identifier: "right_cmd", "fn", "f13", etc.
    #[serde(default = "default_hotkey")]
    pub hotkey: String,
    /// Trigger mode: "toggle" or "hold"
    #[serde(default = "default_trigger_mode")]
    pub trigger_mode: String,
}

fn default_provider() -> String { "local".into() }
fn default_groq_model() -> String { "whisper-large-v3-turbo".into() }
fn default_hotkey() -> String { "right_cmd".into() }
fn default_trigger_mode() -> String { "toggle".into() }
```

Update the `Default` impl accordingly.

- [ ] **Step 2: Add helper method for resolved API key**

```rust
impl Config {
    /// Returns Groq API key: env var GROQ_API_KEY takes precedence over config file.
    pub fn resolved_groq_api_key(&self) -> String {
        std::env::var("GROQ_API_KEY").unwrap_or_else(|_| self.groq_api_key.clone())
    }
}
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check 2>&1 | head -30`
Expected: No errors (existing code only reads `model_path`, serde defaults handle missing fields)

- [ ] **Step 4: Commit**

```bash
git add src/common/config.rs
git commit -m "feat(config): add provider, hotkey, and trigger_mode settings"
```

---

### Task 2: Define AsrProvider trait

**Files:**
- Modify: `src/asr/mod.rs`

- [ ] **Step 1: Add the AsrProvider trait**

Add at the top of `src/asr/mod.rs` (after the existing `use` statements):

```rust
use async_trait::async_trait;

/// Trait abstracting speech recognition backends (local ONNX vs cloud API).
#[async_trait]
pub trait AsrProvider: Send + Sync {
    /// Transcribe PCM audio samples. Returns transcribed text.
    async fn transcribe(&self, audio: &[f32], sample_rate: u32) -> Result<TranscriptionResult>;

    /// Optional warmup (e.g. load model, establish connection). Default no-op.
    async fn warmup(&self) -> Result<()> {
        Ok(())
    }

    /// Check provider readiness. Returns human-readable status string on success.
    fn check_status(&self) -> Result<String> {
        Ok("ready".into())
    }

    /// Provider name for display purposes.
    fn name(&self) -> &str;
}
```

- [ ] **Step 2: Add async-trait dependency**

In `Cargo.toml`, add to `[dependencies]`:
```toml
async-trait = "0.1"
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check 2>&1 | head -30`
Expected: Compiles (trait is defined but not yet used)

- [ ] **Step 4: Commit**

```bash
git add src/asr/mod.rs Cargo.toml Cargo.lock
git commit -m "feat(asr): define AsrProvider trait for pluggable backends"
```

---

### Task 3: Wrap existing AsrEngine as LocalAsrProvider

**Files:**
- Modify: `src/asr/mod.rs`

- [ ] **Step 1: Implement AsrProvider for AsrEngine**

Add at the bottom of `src/asr/mod.rs` (before the `#[cfg(test)]` block):

```rust
/// Local ASR provider wrapping the existing ONNX-based AsrEngine.
/// Uses Arc<Mutex<>> so the engine can be moved into spawn_blocking.
pub struct LocalAsrProvider {
    engine: Arc<Mutex<AsrEngine>>,
}

impl LocalAsrProvider {
    pub fn new(model_path: PathBuf) -> Self {
        Self {
            engine: Arc::new(Mutex::new(AsrEngine::new(model_path))),
        }
    }
}

#[async_trait]
impl AsrProvider for LocalAsrProvider {
    async fn transcribe(&self, audio: &[f32], sample_rate: u32) -> Result<TranscriptionResult> {
        let audio = audio.to_vec();
        let engine = self.engine.clone();
        // ONNX inference is CPU-bound — run in spawn_blocking to avoid
        // blocking the tokio runtime.
        tokio::task::spawn_blocking(move || {
            engine.lock().unwrap().transcribe_pcm(&audio, sample_rate)
        })
        .await
        .context("spawn_blocking failed")?
    }

    async fn warmup(&self) -> Result<()> {
        let engine = self.engine.clone();
        tokio::task::spawn_blocking(move || {
            engine.lock().unwrap().warmup();
        })
        .await
        .context("warmup spawn_blocking failed")?;
        Ok(())
    }

    fn check_status(&self) -> Result<String> {
        let status = self.engine.lock().unwrap().check_status();
        if status.ready {
            Ok("ready".into())
        } else {
            anyhow::bail!(
                "Model not ready: {:?} (onnx={}, model={})",
                status.model_path,
                status.onnx_exists,
                status.model_exists
            )
        }
    }

    fn name(&self) -> &str {
        "local (SenseVoice)"
    }
}
```

Note: `AsrEngine` has `&mut self` on `transcribe_pcm` and `warmup`. The `Mutex` handles interior mutability. We'll need to change `transcribe_pcm` and `warmup` calls to use `&mut self` through the mutex — but `Mutex::lock()` gives `MutexGuard` which derefs to `&mut T` for `lock().unwrap()`. Actually, `Mutex::lock()` returns `MutexGuard<T>` which gives `&mut T` via `DerefMut`. This is fine.

- [ ] **Step 2: Verify it compiles**

Run: `cargo check 2>&1 | head -30`
Expected: Compiles

- [ ] **Step 3: Commit**

```bash
git add src/asr/mod.rs
git commit -m "feat(asr): implement LocalAsrProvider wrapping existing AsrEngine"
```

---

### Task 4: Implement GroqAsrProvider

**Files:**
- Create: `src/asr/groq.rs`
- Modify: `src/asr/mod.rs` (add `pub mod groq;`)
- Modify: `Cargo.toml` (add `multipart` feature to reqwest)

- [ ] **Step 1: Add multipart feature to reqwest in Cargo.toml**

Change the reqwest line:
```toml
reqwest = { version = "0.12", features = ["stream", "rustls-tls", "multipart"], default-features = false }
```

- [ ] **Step 2: Create src/asr/groq.rs**

```rust
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use tracing::info;

use crate::common::types::TranscriptionResult;
use super::AsrProvider;

pub struct GroqAsrProvider {
    api_key: String,
    model: String,
    language: String,
    client: reqwest::Client,
}

impl GroqAsrProvider {
    pub fn new(api_key: String, model: String, language: String) -> Result<Self> {
        if api_key.trim().is_empty() {
            anyhow::bail!("Groq API key is required. Set GROQ_API_KEY env var or groq_api_key in config.toml");
        }
        Ok(Self { api_key, model, language, client: reqwest::Client::new() })
    }
}

#[async_trait]
impl AsrProvider for GroqAsrProvider {
    async fn transcribe(&self, audio: &[f32], sample_rate: u32) -> Result<TranscriptionResult> {
        let start = std::time::Instant::now();
        info!("Groq transcription: {} samples @ {}Hz, model={}", audio.len(), sample_rate, self.model);

        // Encode PCM f32 to WAV in memory
        let wav_bytes = encode_wav(audio, sample_rate)?;

        let file_part = reqwest::multipart::Part::bytes(wav_bytes)
            .file_name("audio.wav")
            .mime_str("audio/wav")?;

        let mut form = reqwest::multipart::Form::new()
            .text("model", self.model.clone())
            .part("file", file_part);

        if !self.language.is_empty() {
            form = form.text("language", self.language.clone());
        }

        let res = self.client
            .post("https://api.groq.com/openai/v1/audio/transcriptions")
            .bearer_auth(&self.api_key)
            .multipart(form)
            .timeout(std::time::Duration::from_secs(30))
            .send()
            .await
            .context("Groq API request failed")?;

        if !res.status().is_success() {
            let status = res.status();
            let body = res.text().await.unwrap_or_default();
            return Err(anyhow!("Groq transcription failed: {} {}", status, body));
        }

        let parsed: GroqTranscriptionResponse = res
            .json()
            .await
            .context("Failed to parse Groq response")?;

        let duration_ms = start.elapsed().as_millis() as u64;
        info!("Groq transcription complete: {}ms, text length={}", duration_ms, parsed.text.len());

        Ok(TranscriptionResult {
            text: parsed.text,
            confidence: 1.0,
            language: if self.language.is_empty() { None } else { Some(self.language.clone()) },
            duration_ms,
        })
    }

    fn check_status(&self) -> Result<String> {
        if self.api_key.trim().is_empty() {
            anyhow::bail!("Groq API key not configured")
        }
        Ok(format!("ready (model: {})", self.model))
    }

    fn name(&self) -> &str {
        "groq (Whisper)"
    }
}

#[derive(serde::Deserialize)]
struct GroqTranscriptionResponse {
    text: String,
}

/// Encode f32 PCM samples to WAV bytes in memory.
fn encode_wav(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    use std::io::Cursor;

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec)
            .context("Failed to create WAV writer")?;
        for &s in samples {
            let clamped = s.clamp(-1.0, 1.0);
            let i16_val = (clamped * 32767.0) as i16;
            writer.write_sample(i16_val).context("Failed to write WAV sample")?;
        }
        writer.finalize().context("Failed to finalize WAV")?;
    }

    Ok(cursor.into_inner())
}
```

- [ ] **Step 3: Add module declaration in src/asr/mod.rs**

Add after the existing `pub mod preprocess;` line:
```rust
pub mod groq;
```

- [ ] **Step 4: Verify it compiles**

Run: `cargo check 2>&1 | head -30`
Expected: Compiles

- [ ] **Step 5: Commit**

```bash
git add src/asr/groq.rs src/asr/mod.rs Cargo.toml Cargo.lock
git commit -m "feat(asr): add GroqAsrProvider for cloud-based Whisper transcription"
```

---

### Task 5: Integrate AsrProvider into Daemon

**Files:**
- Modify: `src/daemon/mod.rs`
- Modify: `src/cli/daemon.rs`

- [ ] **Step 1: Update Daemon to use Arc\<dyn AsrProvider\>**

In `src/daemon/mod.rs`, replace `asr_engine: Arc<Mutex<AsrEngine>>` with `provider: Arc<dyn AsrProvider>`:

Replace the imports at the top:
```rust
use crate::asr::AsrProvider;
```
(Remove `use crate::asr::AsrEngine;`)

Update `Daemon` struct — replace `asr_engine` field:
```rust
provider: Arc<dyn AsrProvider>,
```

Remove the `model_path` field (no longer needed in Daemon — the provider encapsulates it).

Update `Daemon::new` signature:
```rust
pub fn new(
    provider: Arc<dyn AsrProvider>,
    tray: Option<Arc<TrayHandle>>,
) -> Result<Self> {
```

Update constructor body — remove `asr_engine` and `model_path`, add `provider`.

Update `Daemon::run`:
- Replace `self.asr_engine.lock().unwrap().check_status()` block with:
  ```rust
  if let Err(e) = self.provider.check_status() {
      anyhow::bail!("Provider not ready: {}", e);
  }
  ```
- Replace warmup block with:
  ```rust
  {
      let warmup_start = std::time::Instant::now();
      self.provider.warmup().await?;
      let warmup_ms = warmup_start.elapsed().as_millis();
      info!("Provider warmup: {}ms", warmup_ms);
  }
  ```
- Update ready message to show provider name.

Update `stop_and_transcribe`:
- Replace the `spawn_blocking` + `asr_engine.lock().unwrap().transcribe_pcm()` block with:
  ```rust
  let provider = self.provider.clone();
  let buffer_owned = buffer.clone();
  let infer_fut = async move {
      provider.transcribe(&buffer_owned, sample_rate).await
  };
  ```
  Keep the existing 30s timeout wrapper around `infer_fut`.

- [ ] **Step 2: Update run_daemon function**

```rust
pub async fn run_daemon(
    provider: Arc<dyn AsrProvider>,
    tray: Option<Arc<TrayHandle>>,
) -> Result<()> {
    let daemon = Daemon::new(provider, tray)?;
    daemon.run().await
}
```

- [ ] **Step 3: Update cli/daemon.rs start_foreground to construct provider from config**

In `start_foreground`, after model path is resolved, add provider construction:

```rust
use crate::asr::{AsrProvider, LocalAsrProvider};
use crate::asr::groq::GroqAsrProvider;

// After model_path is resolved...
let config = Config::load().unwrap_or_default();
let provider: Arc<dyn AsrProvider> = match config.provider.as_str() {
    "groq" => {
        let api_key = config.resolved_groq_api_key();
        match GroqAsrProvider::new(api_key, config.groq_model.clone(), config.groq_language.clone()) {
            Ok(p) => {
                println!("   Provider: Groq ({})", config.groq_model);
                Arc::new(p)
            }
            Err(e) => {
                eprintln!("⚠️  Groq provider failed: {}. Falling back to local.", e);
                Arc::new(LocalAsrProvider::new(model_path.clone()))
            }
        }
    }
    _ => {
        println!("   Provider: Local (SenseVoice)");
        Arc::new(LocalAsrProvider::new(model_path.clone()))
    }
};
```

Update the `run_daemon` call:
```rust
if let Err(e) = run_daemon(provider, tray_handle).await {
```

- [ ] **Step 4: Verify it compiles**

Run: `cargo check 2>&1 | head -30`
Expected: Compiles

- [ ] **Step 5: Commit**

```bash
git add src/daemon/mod.rs src/cli/daemon.rs
git commit -m "feat(daemon): use AsrProvider trait, support local and groq backends"
```

---

## Chunk 2: Hotkey System Overhaul

### Task 6: Expand HotkeyEvent to Pressed/Released

**Files:**
- Modify: `src/common/types.rs`
- Modify: `src/hotkey/mod.rs`

- [ ] **Step 1: Update HotkeyEvent enum**

In `src/common/types.rs`, replace:
```rust
#[derive(Debug, Clone)]
pub struct HotkeyEvent;
```
with:
```rust
#[derive(Debug, Clone)]
pub enum HotkeyEvent {
    Pressed,
    Released,
}
```

- [ ] **Step 2: Update macOS hotkey listener to send Pressed/Released**

In `src/hotkey/mod.rs`, in `run_listen_loop_macos`:

Change the press branch (line ~88):
```rust
if let Err(e) = sender.send(HotkeyEvent::Pressed) {
```

Add release sending in the release branch (after logging, inside `if was`):
```rust
if was {
    let n = rc.fetch_add(1, Ordering::SeqCst) + 1;
    info!(
        "[Hotkey] event #release={} released (Right Command) was_pressed={}",
        n, was
    );
    if let Err(e) = sender.send(HotkeyEvent::Released) {
        error!("Failed to send hotkey release: {}", e);
    }
}
```

- [ ] **Step 3: Update rdev listener to send Pressed/Released with repeat suppression**

In `run_listen_loop_rdev`, update the `KeyPress` match arm to guard against repeats:
```rust
EventType::KeyPress(Key::MetaRight) => {
    let was = pressed_clone.swap(true, Ordering::SeqCst);
    if !was {
        let n = pc.fetch_add(1, Ordering::SeqCst) + 1;
        info!("[Hotkey] event #press={} pressed", n);
        if let Err(e) = sender.send(HotkeyEvent::Pressed) {
            error!("Failed to send hotkey: {}", e);
        }
    }
}
EventType::KeyRelease(Key::MetaRight) => {
    let was = pressed_clone.swap(false, Ordering::SeqCst);
    if was {
        let n = rc.fetch_add(1, Ordering::SeqCst) + 1;
        info!("[Hotkey] event #release={} released", n);
        if let Err(e) = sender.send(HotkeyEvent::Released) {
            error!("Failed to send hotkey release: {}", e);
        }
    }
}
```

- [ ] **Step 4: Update daemon handle_hotkey for toggle/hold modes**

In `src/daemon/mod.rs`, update `handle_hotkey`:

```rust
async fn handle_hotkey(&self, event: HotkeyEvent, tx: &mpsc::Sender<DaemonEvent>) {
    let n = self.hotkey_recv_count.fetch_add(1, Ordering::SeqCst) + 1;
    let is_processing = self.is_processing.load(Ordering::SeqCst);
    let is_recording = self.state.lock().unwrap().is_recording;
    info!(
        "[Hotkey] #{} event={:?} is_recording={} is_processing={}",
        n, event, is_recording, is_processing
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
                        eprintln!("⚠️  Recording failed: {e}");
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
                        eprintln!("⚠️  Recording failed: {e}");
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
```

Add these fields to `Daemon` struct:
```rust
trigger_mode: String,
provider_name: String,
```

Populate in `Daemon::new()`:
```rust
let config = crate::common::config::Config::load().unwrap_or_default();
// ...
trigger_mode: config.trigger_mode.clone(),
provider_name: provider.name().to_string(),
```

Update the ready message in `Daemon::run` to use `self.provider_name` instead of the removed `self.model_path`:
```rust
println!("   Provider: {}", self.provider_name);
```

- [ ] **Step 5: Verify it compiles**

Run: `cargo check 2>&1 | head -30`
Expected: Compiles

- [ ] **Step 6: Commit**

```bash
git add src/common/types.rs src/hotkey/mod.rs src/daemon/mod.rs
git commit -m "feat(hotkey): support Pressed/Released events with toggle and hold modes"
```

---

### Task 7: Add Fn key listener and configurable hotkey selection

**Files:**
- Modify: `src/hotkey/mod.rs`

- [ ] **Step 1: Add Fn key detection to macOS listener**

Refactor `run_listen_loop_macos` to accept a hotkey config parameter and select the appropriate keycode/flag detection:

Update `HotkeyListener` to accept hotkey config:

```rust
pub struct HotkeyListener {
    sender: Sender<HotkeyEvent>,
    hotkey: String,
}

impl HotkeyListener {
    pub fn new(sender: Sender<HotkeyEvent>, hotkey: String) -> Self {
        Self { sender, hotkey }
    }

    pub fn start(self) -> Result<()> {
        info!("Starting hotkey listener (key: {})...", self.hotkey);
        let hotkey = self.hotkey.clone();
        thread::spawn(move || {
            if let Err(e) = Self::run_listen_loop(self.sender, &hotkey) {
                error!("Hotkey listener error: {}", e);
            }
        });
        Ok(())
    }

    fn run_listen_loop(sender: Sender<HotkeyEvent>, hotkey: &str) -> Result<()> {
        #[cfg(target_os = "macos")]
        {
            return Self::run_listen_loop_macos(sender, hotkey);
        }
        #[cfg(not(target_os = "macos"))]
        {
            return Self::run_listen_loop_rdev(sender, hotkey);
        }
    }
```

- [ ] **Step 2: Update macOS listener to handle "fn" hotkey**

In `run_listen_loop_macos`, add Fn key branch:

```rust
#[cfg(target_os = "macos")]
fn run_listen_loop_macos(sender: Sender<HotkeyEvent>, hotkey: &str) -> Result<()> {
    use core_foundation::runloop::{kCFRunLoopCommonModes, CFRunLoop};
    use core_graphics::event::{
        CGEventFlags, CGEventTap, CGEventTapLocation, CGEventTapOptions,
        CGEventType, EventField, KeyCode,
    };
    use std::sync::atomic::{AtomicBool, AtomicU64};

    let is_fn_key = hotkey == "fn";
    let pressed = Arc::new(AtomicBool::new(false));
    let pressed_clone = pressed.clone();
    let press_count = Arc::new(AtomicU64::new(0));
    let release_count = Arc::new(AtomicU64::new(0));
    let pc = press_count.clone();
    let rc = release_count.clone();

    let key_name = if is_fn_key { "Fn" } else { "Right Command" };
    println!("⌨️  Hotkey listener started (CGEventTap, key: {})", key_name);

    let current = CFRunLoop::get_current();
    let tap = CGEventTap::new(
        CGEventTapLocation::HID,
        core_graphics::event::CGEventTapPlacement::HeadInsertEventTap,
        CGEventTapOptions::ListenOnly,
        vec![CGEventType::FlagsChanged],
        move |_proxy, event_type, event| {
            if event_type as u32 != CGEventType::FlagsChanged as u32 {
                return None;
            }

            let is_pressed;
            if is_fn_key {
                // Fn key: detect via secondary Fn flag (0x800000)
                let flags = event.get_flags();
                is_pressed = (flags.bits() & 0x800000) != 0;
            } else {
                // Right Command: existing logic
                let keycode = event
                    .get_integer_value_field(EventField::KEYBOARD_EVENT_KEYCODE)
                    as u16;
                if keycode != KeyCode::RIGHT_COMMAND {
                    return None;
                }
                is_pressed = event.get_flags().contains(CGEventFlags::CGEventFlagCommand);
            }

            if is_pressed {
                let was = pressed_clone.swap(true, Ordering::SeqCst);
                if !was {
                    let n = pc.fetch_add(1, Ordering::SeqCst) + 1;
                    info!("[Hotkey] #{} pressed ({})", n, if is_fn_key { "Fn" } else { "Right Cmd" });
                    if let Err(e) = sender.send(HotkeyEvent::Pressed) {
                        error!("Failed to send hotkey event: {}", e);
                    }
                }
            } else {
                let was = pressed_clone.swap(false, Ordering::SeqCst);
                if was {
                    let n = rc.fetch_add(1, Ordering::SeqCst) + 1;
                    info!("[Hotkey] #{} released ({})", n, if is_fn_key { "Fn" } else { "Right Cmd" });
                    if let Err(e) = sender.send(HotkeyEvent::Released) {
                        error!("Failed to send hotkey release: {}", e);
                    }
                }
            }

            None
        },
    )
    .map_err(|_| {
        anyhow::anyhow!(
            "CGEventTap creation failed. Please grant Accessibility permission."
        )
    })?;

    let loop_source = tap
        .mach_port
        .create_runloop_source(0)
        .map_err(|_| anyhow::anyhow!("Cannot create CGEventTap RunLoopSource"))?;
    unsafe {
        current.add_source(&loop_source, kCFRunLoopCommonModes);
    }
    tap.enable();
    CFRunLoop::run_current();
    Ok(())
}
```

Note: For the Fn key, we need to handle CGEventFlags differently. The Fn key doesn't produce a specific keycode — it only changes the flags. The `is_fn_key` boolean selects between keycode-based detection (Right Cmd) and flag-based detection (Fn). When Fn is the hotkey, we don't filter by keycode at all — we only check the Fn flag.

- [ ] **Step 3: Update HotkeyListener creation in daemon**

In `src/daemon/mod.rs`, update the listener construction:
```rust
let config = crate::common::config::Config::load().unwrap_or_default();
let listener = HotkeyListener::new(hotkey_tx, config.hotkey.clone());
```

- [ ] **Step 4: Verify it compiles**

Run: `cargo check 2>&1 | head -30`
Expected: Compiles

- [ ] **Step 5: Commit**

```bash
git add src/hotkey/mod.rs src/daemon/mod.rs
git commit -m "feat(hotkey): add Fn key support and configurable hotkey selection"
```

---

## Chunk 3: Floating Indicator Overlay

### Task 8: Create overlay module with NSPanel

**Files:**
- Create: `src/overlay/mod.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Add module declaration**

In `src/lib.rs`, add:
```rust
pub mod overlay;
```

- [ ] **Step 2: Create src/overlay/mod.rs with macOS implementation**

```rust
//! Floating indicator overlay: macOS uses native NSPanel near cursor,
//! other platforms are no-op stubs.

use crate::tray::TrayIconState;

// ─────────────────────────────────────────────────────────────────────────────
// macOS implementation
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
mod platform {
    use super::*;
    use objc::runtime::{Class, Object, BOOL, YES, NO};
    use objc::{class, msg_send, sel, sel_impl};
    use std::sync::atomic::{AtomicBool, Ordering};

    const PILL_WIDTH: f64 = 180.0;
    const PILL_HEIGHT: f64 = 36.0;
    const CURSOR_OFFSET_Y: f64 = 20.0;

    pub struct OverlayWindow {
        panel: *mut Object,
        text_field: *mut Object,
        dot_view: *mut Object,
        visible: AtomicBool,
    }

    unsafe impl Send for OverlayWindow {}
    unsafe impl Sync for OverlayWindow {}

    impl OverlayWindow {
        pub fn new() -> Option<Self> {
            unsafe {
                // NSPanel with borderless style
                let frame = NSRect {
                    origin: NSPoint { x: 0.0, y: 0.0 },
                    size: NSSize { width: PILL_WIDTH, height: PILL_HEIGHT },
                };

                let style_mask: u64 = 0; // NSWindowStyleMaskBorderless
                let panel: *mut Object = msg_send![class!(NSPanel),
                    alloc];
                let panel: *mut Object = msg_send![panel,
                    initWithContentRect: frame
                    styleMask: style_mask
                    backing: 2u64 // NSBackingStoreBuffered
                    defer: NO
                ];

                if panel.is_null() {
                    return None;
                }

                // Configure panel
                let _: () = msg_send![panel, setLevel: 25i64]; // NSStatusWindowLevel
                let _: () = msg_send![panel, setOpaque: NO];
                let _: () = msg_send![panel, setHasShadow: NO];
                let _: () = msg_send![panel, setIgnoresMouseEvents: YES];
                let _: () = msg_send![panel, setCollectionBehavior: 1u64 << 0]; // canJoinAllSpaces

                // Set transparent background
                let clear_color: *mut Object = msg_send![class!(NSColor), clearColor];
                let _: () = msg_send![panel, setBackgroundColor: clear_color];

                // Create content view with rounded dark background
                let content_view: *mut Object = msg_send![panel, contentView];

                // Create a visual effect view for blur
                let effect_view: *mut Object = msg_send![class!(NSVisualEffectView), alloc];
                let content_frame: NSRect = msg_send![content_view, bounds];
                let effect_view: *mut Object = msg_send![effect_view, initWithFrame: content_frame];
                let _: () = msg_send![effect_view, setMaterial: 13i64]; // NSVisualEffectMaterialHUDWindow
                let _: () = msg_send![effect_view, setBlendingMode: 0i64]; // behindWindow
                let _: () = msg_send![effect_view, setState: 1i64]; // active
                let _: () = msg_send![effect_view, setWantsLayer: YES];

                // Round corners
                let layer: *mut Object = msg_send![effect_view, layer];
                if !layer.is_null() {
                    let _: () = msg_send![layer, setCornerRadius: (PILL_HEIGHT / 2.0) as f64];
                    let _: () = msg_send![layer, setMasksToBounds: YES];
                }

                let _: () = msg_send![content_view, addSubview: effect_view];

                // Red dot view
                let dot_frame = NSRect {
                    origin: NSPoint { x: 14.0, y: (PILL_HEIGHT - 10.0) / 2.0 },
                    size: NSSize { width: 10.0, height: 10.0 },
                };
                let dot_view: *mut Object = msg_send![class!(NSView), alloc];
                let dot_view: *mut Object = msg_send![dot_view, initWithFrame: dot_frame];
                let _: () = msg_send![dot_view, setWantsLayer: YES];
                let dot_layer: *mut Object = msg_send![dot_view, layer];
                if !dot_layer.is_null() {
                    let _: () = msg_send![dot_layer, setCornerRadius: 5.0f64];
                    let red: *mut Object = msg_send![class!(NSColor), redColor];
                    let cg_red: *mut Object = msg_send![red, CGColor];
                    let _: () = msg_send![dot_layer, setBackgroundColor: cg_red];
                }
                let _: () = msg_send![effect_view, addSubview: dot_view];

                // Add pulsing animation to dot
                Self::add_pulse_animation(dot_layer);

                // Text label
                let text_frame = NSRect {
                    origin: NSPoint { x: 32.0, y: 0.0 },
                    size: NSSize { width: PILL_WIDTH - 44.0, height: PILL_HEIGHT },
                };
                let text_field: *mut Object = msg_send![class!(NSTextField), alloc];
                let text_field: *mut Object = msg_send![text_field, initWithFrame: text_frame];
                let _: () = msg_send![text_field, setBezeled: NO];
                let _: () = msg_send![text_field, setDrawsBackground: NO];
                let _: () = msg_send![text_field, setEditable: NO];
                let _: () = msg_send![text_field, setSelectable: NO];

                let white: *mut Object = msg_send![class!(NSColor), whiteColor];
                let _: () = msg_send![text_field, setTextColor: white];

                let font: *mut Object = msg_send![class!(NSFont), systemFontOfSize: 13.0f64];
                let _: () = msg_send![text_field, setFont: font];

                let ns_str = Self::ns_string("Recording...");
                let _: () = msg_send![text_field, setStringValue: ns_str];

                let _: () = msg_send![effect_view, addSubview: text_field];

                // Start hidden
                let _: () = msg_send![panel, orderOut: std::ptr::null::<Object>()];

                Some(Self {
                    panel,
                    text_field,
                    dot_view,
                    visible: AtomicBool::new(false),
                })
            }
        }

        unsafe fn add_pulse_animation(layer: *mut Object) {
            if layer.is_null() {
                return;
            }
            let anim: *mut Object = msg_send![class!(CABasicAnimation),
                animationWithKeyPath: Self::ns_string("opacity")];
            let from: *mut Object = msg_send![class!(NSNumber), numberWithFloat: 1.0f32];
            let to: *mut Object = msg_send![class!(NSNumber), numberWithFloat: 0.3f32];
            let _: () = msg_send![anim, setFromValue: from];
            let _: () = msg_send![anim, setToValue: to];
            let _: () = msg_send![anim, setDuration: 0.8f64];
            let _: () = msg_send![anim, setAutoreverses: YES];
            let _: () = msg_send![anim, setRepeatCount: f32::MAX];
            let key = Self::ns_string("pulse");
            let _: () = msg_send![layer, addAnimation: anim forKey: key];
        }

        unsafe fn ns_string(s: &str) -> *mut Object {
            let cls = class!(NSString);
            let c_str = std::ffi::CString::new(s).unwrap();
            msg_send![cls,
                stringWithUTF8String: c_str.as_ptr()]
        }

        pub fn update_state(&self, state: TrayIconState) {
            unsafe {
                match state {
                    TrayIconState::Idle => {
                        if self.visible.swap(false, Ordering::SeqCst) {
                            let _: () = msg_send![self.panel, orderOut: std::ptr::null::<Object>()];
                        }
                    }
                    TrayIconState::Recording => {
                        // Position near cursor
                        self.position_near_cursor();
                        let ns_str = Self::ns_string("Recording...");
                        let _: () = msg_send![self.text_field, setStringValue: ns_str];
                        // Red dot
                        let dot_layer: *mut Object = msg_send![self.dot_view, layer];
                        if !dot_layer.is_null() {
                            let red: *mut Object = msg_send![class!(NSColor), redColor];
                            let cg_red: *mut Object = msg_send![red, CGColor];
                            let _: () = msg_send![dot_layer, setBackgroundColor: cg_red];
                        }
                        if !self.visible.swap(true, Ordering::SeqCst) {
                            let _: () = msg_send![self.panel, makeKeyAndOrderFront: std::ptr::null::<Object>()];
                        }
                    }
                    TrayIconState::Transcribing => {
                        let ns_str = Self::ns_string("Transcribing...");
                        let _: () = msg_send![self.text_field, setStringValue: ns_str];
                        // Orange dot
                        let dot_layer: *mut Object = msg_send![self.dot_view, layer];
                        if !dot_layer.is_null() {
                            let orange: *mut Object = msg_send![class!(NSColor), orangeColor];
                            let cg_orange: *mut Object = msg_send![orange, CGColor];
                            let _: () = msg_send![dot_layer, setBackgroundColor: cg_orange];
                        }
                        if !self.visible.load(Ordering::SeqCst) {
                            self.visible.store(true, Ordering::SeqCst);
                            let _: () = msg_send![self.panel, makeKeyAndOrderFront: std::ptr::null::<Object>()];
                        }
                    }
                }
            }
        }

        unsafe fn position_near_cursor(&self) {
            // Get cursor position
            let mouse_loc: NSPoint = msg_send![class!(NSEvent), mouseLocation];

            // Find the screen containing the cursor
            let screens: *mut Object = msg_send![class!(NSScreen), screens];
            let screen_count: u64 = msg_send![screens, count];
            let mut target_frame = NSRect {
                origin: NSPoint { x: 0.0, y: 0.0 },
                size: NSSize { width: 1920.0, height: 1080.0 },
            };

            for i in 0..screen_count {
                let screen: *mut Object = msg_send![screens, objectAtIndex: i];
                let frame: NSRect = msg_send![screen, frame];
                if mouse_loc.x >= frame.origin.x
                    && mouse_loc.x <= frame.origin.x + frame.size.width
                    && mouse_loc.y >= frame.origin.y
                    && mouse_loc.y <= frame.origin.y + frame.size.height
                {
                    target_frame = frame;
                    break;
                }
            }

            // Position below cursor, centered, clamped to screen
            let mut x = mouse_loc.x - PILL_WIDTH / 2.0;
            let mut y = mouse_loc.y - PILL_HEIGHT - CURSOR_OFFSET_Y;

            // Clamp to screen bounds
            x = x.max(target_frame.origin.x)
                .min(target_frame.origin.x + target_frame.size.width - PILL_WIDTH);
            y = y.max(target_frame.origin.y)
                .min(target_frame.origin.y + target_frame.size.height - PILL_HEIGHT);

            let origin = NSPoint { x, y };
            let _: () = msg_send![self.panel, setFrameOrigin: origin];
        }
    }

    impl Drop for OverlayWindow {
        fn drop(&mut self) {
            unsafe {
                let _: () = msg_send![self.panel, orderOut: std::ptr::null::<Object>()];
                let _: () = msg_send![self.panel, close];
            }
        }
    }

    // NSRect/NSPoint/NSSize for FFI
    #[repr(C)]
    #[derive(Copy, Clone)]
    struct NSPoint { x: f64, y: f64 }
    #[repr(C)]
    #[derive(Copy, Clone)]
    struct NSSize { width: f64, height: f64 }
    #[repr(C)]
    #[derive(Copy, Clone)]
    struct NSRect { origin: NSPoint, size: NSSize }
}

// ─────────────────────────────────────────────────────────────────────────────
// Non-macOS stub
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(not(target_os = "macos"))]
mod platform {
    use super::*;

    pub struct OverlayWindow;

    impl OverlayWindow {
        pub fn new() -> Option<Self> {
            Some(Self)
        }

        pub fn update_state(&self, _state: TrayIconState) {}
    }
}

pub use platform::OverlayWindow;
```

- [ ] **Step 3: Integrate overlay into main loop**

In `src/cli/daemon.rs`, create the overlay on the main thread (after tray creation) and flush state updates alongside the tray:

After `TrayState::new()`, add:
```rust
use crate::overlay::OverlayWindow;

let overlay = OverlayWindow::new();
if overlay.is_some() {
    tracing::info!("✅ Overlay window created");
}
```

The overlay state updates need the same channel as the tray. Add a second `SyncSender<TrayIconState>` for the overlay, or have the main loop drive overlay updates from the same tray state channel.

Simplest approach: In `run_main_loop`, after `flush_state_updates()`, also update the overlay. Pass the overlay into `run_main_loop`:

```rust
fn run_main_loop(tray: Option<&TrayState>, overlay: Option<&OverlayWindow>) {
```

In the loop, after tray flush:
```rust
// Overlay mirrors tray state — the tray's apply_state already runs,
// but we need to feed the same states to overlay.
// We'll add a secondary receiver for overlay.
```

Actually, a cleaner approach: Add a second `sync_channel` to `TrayHandle` for the overlay. When `TrayHandle::set_state` is called, it sends to both channels. Then in the main loop, drain the overlay channel too:

In `src/tray/mod.rs` (macOS platform), add an overlay state sender to `TrayHandle`:
```rust
pub struct TrayHandle {
    pub(super) state_tx: std::sync::mpsc::SyncSender<TrayIconState>,
    pub(super) overlay_state_tx: Option<std::sync::mpsc::SyncSender<TrayIconState>>,
    pub(super) exit_requested: Arc<AtomicBool>,
}

impl TrayHandle {
    pub fn set_state(&self, state: TrayIconState) {
        let _ = self.state_tx.try_send(state);
        if let Some(ref tx) = self.overlay_state_tx {
            let _ = tx.try_send(state);
        }
    }

    pub fn set_overlay_sender(&mut self, tx: std::sync::mpsc::SyncSender<TrayIconState>) {
        self.overlay_state_tx = Some(tx);
    }
}
```

In `cli/daemon.rs`, create the overlay channel and attach it BEFORE wrapping in Arc:
```rust
let (overlay_tx, overlay_rx) = std::sync::mpsc::sync_channel::<TrayIconState>(16);
// Set overlay sender before wrapping in Arc to avoid Arc::get_mut issues
if let Some(ref mut handle) = tray_handle_raw {
    handle.set_overlay_sender(overlay_tx);
}
let tray_handle = tray_handle_raw.map(Arc::new);
```
(This requires deferring the `Arc::new` wrapping from where it currently happens in the tray creation block.)

Then in `run_main_loop`, drain `overlay_rx` and call `overlay.update_state()`.

- [ ] **Step 4: Verify it compiles**

Run: `cargo check 2>&1 | head -30`
Expected: Compiles

- [ ] **Step 5: Commit**

```bash
git add src/overlay/mod.rs src/lib.rs src/tray/mod.rs src/cli/daemon.rs
git commit -m "feat(overlay): add native macOS floating indicator near cursor"
```

---

## Chunk 4: Settings Window

### Task 9: Create settings window module

**Files:**
- Create: `src/settings_window/mod.rs`
- Modify: `src/lib.rs`
- Modify: `src/tray/mod.rs`

- [ ] **Step 1: Add module declaration in lib.rs**

```rust
pub mod settings_window;
```

- [ ] **Step 2: Create src/settings_window/mod.rs**

This is the most complex native UI piece. It creates an NSWindow with native macOS controls for:
- Provider segmented control (Local / Groq)
- Groq API key text field
- Groq model dropdown
- Hotkey button + Fn preset
- Trigger mode segmented control (Toggle / Hold)

Due to the complexity of native Cocoa UI from Rust, this will be implemented as a single NSWindow with manually laid out controls using `objc` message sends. The full implementation is substantial (300+ lines of unsafe Cocoa code).

Key architectural points:
- Window is created once, shown/hidden on demand
- Controls read from and write to `Config` on every interaction
- Hotkey changes signal the daemon to restart the hotkey listener (via a shared `Arc<AtomicBool>` "config changed" flag)
- The window runs on the main NSRunLoop thread

```rust
//! Settings window: macOS has native NSWindow, other platforms are no-op stubs.

// ─────────────────────────────────────────────────────────────────────────────
// macOS implementation
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
mod platform {
    use objc::runtime::{Object, BOOL, YES, NO};
    use objc::{class, msg_send, sel, sel_impl};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    const WINDOW_WIDTH: f64 = 400.0;
    const WINDOW_HEIGHT: f64 = 320.0;

    pub struct SettingsWindow {
        window: *mut Object,
        visible: AtomicBool,
    }

    unsafe impl Send for SettingsWindow {}
    unsafe impl Sync for SettingsWindow {}

    impl SettingsWindow {
        pub fn new() -> Option<Self> {
            unsafe {
                let frame = NSRect {
                    origin: NSPoint { x: 200.0, y: 200.0 },
                    size: NSSize { width: WINDOW_WIDTH, height: WINDOW_HEIGHT },
                };

                // NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
                let style_mask: u64 = (1 << 0) | (1 << 1);
                let window: *mut Object = msg_send![class!(NSWindow), alloc];
                let window: *mut Object = msg_send![window,
                    initWithContentRect: frame
                    styleMask: style_mask
                    backing: 2u64
                    defer: NO
                ];

                if window.is_null() {
                    return None;
                }

                let title = Self::ns_string("Open Flow Preferences");
                let _: () = msg_send![window, setTitle: title];
                let _: () = msg_send![window, center];

                // Build the settings controls
                Self::build_ui(window);

                // Start hidden
                let _: () = msg_send![window, orderOut: std::ptr::null::<Object>()];

                Some(Self {
                    window,
                    visible: AtomicBool::new(false),
                })
            }
        }

        pub fn show(&self) {
            unsafe {
                // Reload config values into controls before showing
                self.reload_values();
                let _: () = msg_send![self.window, makeKeyAndOrderFront: std::ptr::null::<Object>()];
                // Activate app so window can receive focus (LSUIElement apps need this)
                let app: *mut Object = msg_send![class!(NSApplication), sharedApplication];
                let _: () = msg_send![app, activateIgnoringOtherApps: YES];
                self.visible.store(true, Ordering::SeqCst);
            }
        }

        pub fn is_visible(&self) -> bool {
            self.visible.load(Ordering::SeqCst)
        }

        unsafe fn build_ui(window: *mut Object) {
            let content: *mut Object = msg_send![window, contentView];
            let mut y = WINDOW_HEIGHT - 50.0;
            let label_x = 20.0;
            let control_x = 150.0;
            let control_width = WINDOW_WIDTH - control_x - 20.0;
            let row_height = 36.0;

            // --- Provider ---
            Self::add_label(content, "Provider:", label_x, y);
            // For now, add a simple label showing current provider.
            // Full NSSegmentedControl + callbacks would require Objective-C delegate
            // pattern from Rust. We'll use a simplified approach: each control is
            // tagged and we poll their values when the window is about to close.
            // This is a pragmatic approach for v1.
            Self::add_label(content, "(Edit config.toml to change settings)", label_x, y - row_height);

            // The full native UI with interactive controls will be implemented
            // iteratively. For v1, the settings window shows current values
            // and provides a button to open config.toml in the default editor.

            y -= row_height * 3.0;
            Self::add_label(content, "Current settings will be shown here.", label_x, y);

            y -= row_height * 2.0;

            // "Open Config File" button
            let btn_frame = NSRect {
                origin: NSPoint { x: (WINDOW_WIDTH - 160.0) / 2.0, y: 20.0 },
                size: NSSize { width: 160.0, height: 32.0 },
            };
            let button: *mut Object = msg_send![class!(NSButton), alloc];
            let button: *mut Object = msg_send![button, initWithFrame: btn_frame];
            let btn_title = Self::ns_string("Open Config File");
            let _: () = msg_send![button, setTitle: btn_title];
            let _: () = msg_send![button, setBezelStyle: 1i64]; // rounded
            // TODO: Add action target to open config file
            let _: () = msg_send![content, addSubview: button];
        }

        unsafe fn add_label(parent: *mut Object, text: &str, x: f64, y: f64) {
            let frame = NSRect {
                origin: NSPoint { x, y },
                size: NSSize { width: 360.0, height: 20.0 },
            };
            let label: *mut Object = msg_send![class!(NSTextField), alloc];
            let label: *mut Object = msg_send![label, initWithFrame: frame];
            let _: () = msg_send![label, setBezeled: NO];
            let _: () = msg_send![label, setDrawsBackground: NO];
            let _: () = msg_send![label, setEditable: NO];
            let _: () = msg_send![label, setSelectable: NO];
            let ns_str = Self::ns_string(text);
            let _: () = msg_send![label, setStringValue: ns_str];
            let _: () = msg_send![parent, addSubview: label];
        }

        fn reload_values(&self) {
            // Load current config and update control values
            // For v1, this is a no-op since controls are static labels
        }

        unsafe fn ns_string(s: &str) -> *mut Object {
            let cls = class!(NSString);
            msg_send![cls,
                stringWithUTF8String: s.as_bytes().as_ptr() as *const std::os::raw::c_char]
        }
    }

    impl Drop for SettingsWindow {
        fn drop(&mut self) {
            unsafe {
                let _: () = msg_send![self.window, orderOut: std::ptr::null::<Object>()];
                let _: () = msg_send![self.window, close];
            }
        }
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct NSPoint { x: f64, y: f64 }
    #[repr(C)]
    #[derive(Copy, Clone)]
    struct NSSize { width: f64, height: f64 }
    #[repr(C)]
    #[derive(Copy, Clone)]
    struct NSRect { origin: NSPoint, size: NSSize }
}

// ─────────────────────────────────────────────────────────────────────────────
// Non-macOS stub
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(not(target_os = "macos"))]
mod platform {
    pub struct SettingsWindow;

    impl SettingsWindow {
        pub fn new() -> Option<Self> { Some(Self) }
        pub fn show(&self) {}
        pub fn is_visible(&self) -> bool { false }
    }
}

pub use platform::SettingsWindow;
```

Note: The v1 settings window is intentionally simple — it shows labels and a button to open the config file. Full interactive controls (segmented controls, text fields with callbacks) require Objective-C delegate/action patterns from Rust which are complex. This can be enhanced iteratively in follow-up work.

- [ ] **Step 3: Add "Preferences..." menu item to tray**

In `src/tray/mod.rs` (macOS platform module), update `TrayState::new()`:

Add a new `MenuItem` for preferences:
```rust
let prefs = MenuItem::with_id("prefs", "Preferences...", true, None);
```

Update the menu:
```rust
let menu = Menu::with_items(&[&title, &status_item, &prefs, &exit])
```

Add handling in `flush_menu_events`:
```rust
} else if event.id.as_ref() == "prefs" {
    info!("User clicked Preferences menu");
    self.prefs_requested.store(true, Ordering::SeqCst);
}
```

Add `prefs_requested: Arc<AtomicBool>` to `TrayState` and `TrayHandle`, with a method `prefs_requested() -> bool` and `clear_prefs_request()`.

- [ ] **Step 4: Integrate into main loop**

In `cli/daemon.rs`, create settings window and show it when tray prefs is requested:

```rust
let settings_window = SettingsWindow::new();

// In run_main_loop:
if tray.map_or(false, |t| t.prefs_requested()) {
    t.clear_prefs_request();
    if let Some(ref sw) = settings_window {
        sw.show();
    }
}
```

- [ ] **Step 5: Verify it compiles**

Run: `cargo check 2>&1 | head -30`
Expected: Compiles

- [ ] **Step 6: Commit**

```bash
git add src/settings_window/mod.rs src/lib.rs src/tray/mod.rs src/cli/daemon.rs
git commit -m "feat(settings): add native macOS settings window with Preferences menu"
```

---

### Task 10: Update .app bundle Info.plist

**Files:**
- Modify: `scripts/build-app.sh`

- [ ] **Step 1: Add accessibility description and update LSUIElement**

In `scripts/build-app.sh`, update the Info.plist heredoc:

Add after `NSSpeechRecognitionUsageDescription`:
```xml
	<key>NSAccessibilityUsageDescription</key>
	<string>Open Flow needs accessibility permission to detect global hotkeys and inject text.</string>
```

Change `LSUIElement` from `false` to `true` (menu-bar-only app, no Dock icon):
```xml
	<key>LSUIElement</key>
	<true/>
```

- [ ] **Step 2: Commit**

```bash
git add scripts/build-app.sh
git commit -m "fix(app): add accessibility description, set LSUIElement=true for menu-bar app"
```

---

### Task 11: Build and smoke test

- [ ] **Step 1: Build release**

Run: `cargo build --release 2>&1 | tail -20`
Expected: Compiles successfully

- [ ] **Step 2: Run tests**

Run: `cargo test 2>&1`
Expected: All existing tests pass

- [ ] **Step 3: Test with local provider**

Run: `./target/release/open-flow start --foreground`
Expected: Starts with tray icon, overlay, "Provider: Local (SenseVoice)" in output

- [ ] **Step 4: Test config changes**

Edit config.toml to set `hotkey = "fn"` and `trigger_mode = "hold"`, restart daemon.
Expected: Fn key triggers recording, holding = recording, release = transcribe.

- [ ] **Step 5: Test Groq provider (if API key available)**

Set `GROQ_API_KEY=...` env var, edit config to `provider = "groq"`, restart.
Expected: Groq provider initializes, transcription uses Groq API.

---

## Known Gaps (v1 → v2)

These items are in the spec but deferred to a follow-up iteration:

1. **Live config reload**: Changes currently require daemon restart. Spec calls for `Arc<RwLock<Config>>` shared between settings window and daemon. Deferred because hotkey listener (CGEventTap callback) captures values at creation time and requires full restart anyway.
2. **Full interactive settings window**: v1 shows current values and an "Open Config File" button. Spec calls for `NSSegmentedControl`, `NSSecureTextField`, `NSPopUpButton`, and hotkey capture mode. Deferred due to complexity of Cocoa delegate/action patterns from Rust.
3. **Hotkey picker "press to set"**: Requires the full interactive settings window.
4. **macOS Keychain for API key**: Config stores API key in plaintext. Spec mentions Keychain as future improvement.

## Notes for Implementers

- **New directories needed**: `src/overlay/` and `src/settings_window/` must be created before writing `mod.rs` files. Use `mkdir -p src/overlay src/settings_window`.
- **NSPoint/NSSize/NSRect**: The overlay and settings_window modules define their own `#[repr(C)]` FFI structs for these. Consider using `cocoa::foundation::{NSPoint, NSSize, NSRect}` from the existing `cocoa` crate dependency instead, but verify the types are re-exported and compatible.
- **Intermediate compilation**: Between Chunk 1 and Chunk 2, `HotkeyListener::new` still uses the old single-argument constructor. Each chunk compiles independently, but you cannot partially apply Chunk 2.

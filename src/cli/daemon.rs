use anyhow::{Context, Result};
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::asr::groq::GroqAsrProvider;
use crate::asr::{AsrProvider, LocalAsrProvider};
use crate::common::config::Config;
use crate::daemon::run_daemon;
use crate::tray::{TrayIconState, TrayState};

/// Set to true when SIGTERM/SIGINT is received, main loop checks and exits gracefully
static SIGNAL_SHUTDOWN: AtomicBool = AtomicBool::new(false);

// ─────────────────────────────────────────────────────────────────────────────
// Common path helpers
// ─────────────────────────────────────────────────────────────────────────────

fn pid_path() -> Result<PathBuf> {
    Ok(Config::data_dir()?.join("daemon.pid"))
}

fn log_path() -> Result<PathBuf> {
    Ok(Config::data_dir()?.join("daemon.log"))
}

/// Read PID file, return pid (returns None if file doesn't exist or content is invalid)
fn read_pid() -> Option<u32> {
    let path = pid_path().ok()?;
    let s = fs::read_to_string(path).ok()?;
    s.trim().parse::<u32>().ok()
}

/// Check if process exists (Unix: kill(pid,0); Windows: OpenProcess + GetExitCodeProcess)
fn is_running(pid: u32) -> bool {
    #[cfg(unix)]
    {
        if unsafe { libc::kill(pid as libc::pid_t, 0) } != 0 {
            return false;
        }
        // Verify it's actually an open-flow process, not a recycled PID
        if let Ok(output) = std::process::Command::new("ps")
            .args(["-p", &pid.to_string(), "-o", "comm="])
            .output()
        {
            let comm = String::from_utf8_lossy(&output.stdout);
            return comm.trim().contains("open-flow");
        }
        true // if ps fails, assume it's ours
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
        use windows_sys::Win32::System::Threading::{
            GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
        };
        const STILL_ACTIVE: u32 = 259;
        let h = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
        if h == 0 || h == -1_i32 as isize {
            return false;
        }
        let mut code: u32 = 0;
        let ok = unsafe { GetExitCodeProcess(h as HANDLE, &mut code) != 0 };
        unsafe { CloseHandle(h as HANDLE) };
        ok && code == STILL_ACTIVE
    }
}

/// Delete PID file (ignore errors)
fn remove_pid_file() {
    if let Ok(p) = pid_path() {
        let _ = fs::remove_file(p);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// start: background by default / optional foreground
// ─────────────────────────────────────────────────────────────────────────────

/// Background start: spawn child process to run daemon, parent exits immediately; closing terminal doesn't affect child.
pub fn start_background(model: Option<PathBuf>) -> anyhow::Result<()> {
    if let Some(pid) = read_pid() {
        if is_running(pid) {
            println!("ℹ️  Open Flow is already running (PID: {})", pid);
            println!("   Stop: open-flow stop");
            return Ok(());
        }
        remove_pid_file();
    }

    let exe = std::env::current_exe().context("Cannot get executable path")?;
    let log = log_path()?;
    fs::create_dir_all(log.parent().unwrap())?;

    let mut args = vec!["start".to_string(), "--foreground".to_string()];
    if let Some(ref m) = model {
        args.push("--model".to_string());
        args.push(m.display().to_string());
    }

    let log_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log)
        .context("Cannot open log file")?;

    let child = Command::new(&exe)
        .args(&args)
        .env("OPEN_FLOW_DAEMON", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file.try_clone()?))
        .stderr(Stdio::from(log_file))
        .spawn()
        .context("Failed to start background process")?;

    let pid = child.id();
    // Parent writes PID file immediately so stop command doesn't need to wait for child readiness
    fs::write(pid_path()?, pid.to_string()).context("Failed to write PID file")?;

    println!("✅ Open Flow started in background (PID: {})", pid);
    println!("   Log: {}", log.display());
    println!("   Stop: open-flow stop");
    Ok(())
}

/// Foreground start: terminal is occupied, Ctrl+C or tray "Exit" to stop.
/// Main thread drives macOS NSRunLoop (tray events), tokio runs background thread (recording/transcription/hotkey).
pub fn start_foreground(model: Option<PathBuf>) -> anyhow::Result<()> {
    // ── Check if already running ─────────────────────────────────────────────────
    if let Some(pid) = read_pid() {
        if is_running(pid) {
            println!("ℹ️  Open Flow is already running (PID: {})", pid);
            println!("   Stop: open-flow stop");
            return Ok(());
        }
        remove_pid_file();
    }

    // ── Temporary tokio runtime (only for model download) ────────────────────────────
    let rt_temp = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("Cannot create tokio runtime")?;

    // ── Model readiness (auto-download on first run, async, executed with block_on) ──────────────
    let model_path = rt_temp
        .block_on(crate::cli::commands::setup::ensure_model_ready(model))
        .map_err(|e| {
            eprintln!("❌ Model preparation failed: {}", e);
            e
        })?;

    // ── Persist model path to config file (for status command display) ─────────────
    if let Ok(mut config) = Config::load() {
        config.model_path = Some(model_path.clone());
        let _ = config.save();
    }

    // ── Write PID file (written when calling --foreground directly; parent already wrote for background start) ────
    let my_pid = std::process::id();
    let _ = fs::write(pid_path()?, my_pid.to_string());

    // ── Register Ctrl+C / signal handling -> set flag, main loop exits gracefully ─────────────────
    #[cfg(not(windows))]
    {
        // Unix: SIGINT/SIGTERM
        let _ = ctrlc::set_handler(move || {
            SIGNAL_SHUTDOWN.store(true, Ordering::SeqCst);
        });
    }
    #[cfg(windows)]
    {
        // Windows: SetConsoleCtrlHandler must return TRUE(1) to indicate handled, otherwise process will be terminated by the system
        use windows_sys::Win32::System::Console::SetConsoleCtrlHandler;
        unsafe {
            let handler = Some(win32_ctrl_handler as _);
            SetConsoleCtrlHandler(handler, 1i32); // 1 = TRUE = add handler
        }
    }

    // ── Initialize AppKit / NSApplication first, then create tray ────────────────────
    // tray-icon on macOS requires the main thread event loop to have started processing events before creating TrayIcon,
    // otherwise the status icon may not display at all.
    #[cfg(target_os = "macos")]
    {
        prepare_appkit();
        pump_run_loop_100ms();
    }

    // ── Create tray on main thread (macOS requires NSStatusItem on main thread; other platforms are stubs) ──────
    let (mut tray, mut tray_handle_raw) = match TrayState::new() {
        Ok((t, h)) => {
            t.set_state(TrayIconState::Idle);
            tracing::info!("✅ Tray icon created");
            (Some(t), Some(h))
        }
        Err(e) => {
            tracing::warn!("Tray icon creation failed: {}, continuing without tray", e);
            (None, None)
        }
    };

    // ── Create floating indicator (overlay) ──────────────────────────────────────
    use crate::overlay::OverlayWindow;
    let overlay = OverlayWindow::new();
    let (overlay_tx, overlay_rx) = std::sync::mpsc::sync_channel::<TrayIconState>(16);
    // Set overlay sender before wrapping in Arc
    if let Some(ref mut handle) = tray_handle_raw {
        handle.set_overlay_sender(overlay_tx);
    }
    let tray_handle = tray_handle_raw.map(Arc::new);
    if overlay.is_some() {
        tracing::info!("✅ Overlay window created");
    }

    // ── Settings app path (bundled alongside main binary) ──────────
    let settings_app_path = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|p| p.join("OpenFlowSettings")))
        .filter(|p| p.exists());

    // ── Build ASR provider ──────────────────────────────────────────
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

    // ── Run daemon on dedicated thread (current_thread runtime, Daemon contains cpal::Stream which is not Send)
    let log = log_path()?;
    println!("✅ Open Flow started (PID: {})", my_pid);
    println!("   Hotkey: {}", config.hotkey);
    println!("   Trigger: {}", config.trigger_mode);
    println!("   Log: {}", log.display());
    println!();
    println!("   Press Ctrl+C or tray menu \"Exit\" to stop");
    println!("   ⏳ Model loading and warmup takes ~3-5 seconds, hotkey will be available after completion");

    let daemon_alive = Arc::new(AtomicBool::new(true));
    let daemon_alive_clone = daemon_alive.clone();

    let daemon_handle = std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("daemon tokio runtime");
        rt.block_on(async {
            if let Err(e) = run_daemon(provider, tray_handle).await {
                eprintln!("Daemon error: {}", e);
            }
        });
        daemon_alive_clone.store(false, Ordering::SeqCst);
    });

    // ── Main thread: drive macOS NSRunLoop to dispatch tray / menu events ──────
    run_main_loop(tray.as_ref(), overlay.as_ref(), &overlay_rx, settings_app_path.as_deref(), &daemon_alive);

    // ── Explicitly hide menu bar icon and pump run loop before exit, to avoid icon remnants
    if let Some(ref t) = tray {
        t.hide_from_menu_bar();
    }
    drop(tray.take());
    #[cfg(target_os = "macos")]
    for _ in 0..10 {
        pump_run_loop_100ms();
    }

    // ── Exit cleanup ──────────────────────────────────────────────────────
    remove_pid_file();

    // Kill the settings app if it's running (it's inside the .app bundle,
    // keeping it alive prevents replacing the bundle)
    let _ = std::process::Command::new("pkill")
        .args(["-x", "OpenFlowSettings"])
        .output();

    // Give the daemon thread a short time to exit gracefully
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while std::time::Instant::now() < deadline {
        if daemon_handle.is_finished() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    println!("\n👋 Open Flow stopped");

    // Force exit — the daemon thread may be blocked on tokio recv() or CFRunLoop
    std::process::exit(0);
}

/// macOS main loop: run NSRunLoop every 100ms, check if exit is needed.
/// This is key to making tray-icon render and respond to menus properly on macOS.
fn run_main_loop(
    tray: Option<&TrayState>,
    overlay: Option<&crate::overlay::OverlayWindow>,
    overlay_rx: &std::sync::mpsc::Receiver<TrayIconState>,
    settings_app_path: Option<&std::path::Path>,
    daemon_alive: &AtomicBool,
) {
    loop {
        // Autorelease pool: every ObjC call in this loop creates autoreleased objects
        // (NSEvent, NSString, etc). Without draining, they accumulate forever → memory leak.
        #[cfg(target_os = "macos")]
        let _pool = unsafe {
            use objc::{class, msg_send, sel, sel_impl};
            let pool: *mut objc::runtime::Object =
                msg_send![class!(NSAutoreleasePool), new];
            pool
        };

        // Apply tray state updates from daemon (gray/red/yellow)
        if let Some(t) = tray {
            t.flush_state_updates();
            t.flush_menu_events();
        }

        // Apply overlay state updates
        while let Ok(state) = overlay_rx.try_recv() {
            if let Some(o) = overlay {
                o.update_state(state);
            }
        }

        // Drive platform event loop
        #[cfg(target_os = "macos")]
        pump_run_loop_100ms();

        #[cfg(target_os = "linux")]
        {
            pump_glib_linux();
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        #[cfg(target_os = "windows")]
        {
            pump_win32_messages();
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Tray menu "Preferences..." -> launch SwiftUI settings app
        if tray.map_or(false, |t| t.prefs_requested()) {
            if let Some(path) = settings_app_path {
                let _ = std::process::Command::new(path).spawn();
            } else {
                tracing::warn!("Settings app not found alongside binary");
            }
        }

        // Check exit conditions
        let should_exit = if tray.map_or(false, |t| t.exit_requested()) {
            tracing::info!("User clicked tray exit");
            true
        } else if SIGNAL_SHUTDOWN.load(Ordering::SeqCst) {
            tracing::info!("Signal received, exiting gracefully");
            true
        } else if !daemon_alive.load(Ordering::SeqCst) {
            tracing::error!("Daemon thread has exited unexpectedly");
            true
        } else {
            false
        };

        // Drain autorelease pool — MUST happen every iteration to prevent memory leak
        #[cfg(target_os = "macos")]
        unsafe {
            use objc::{msg_send, sel, sel_impl};
            let _: () = msg_send![_pool, drain];
        }

        if should_exit {
            break;
        }
    }
}

/// Drive the AppKit event queue via `[NSApp nextEventMatchingMask:...]`.
/// NSRunLoop::runUntilDate only handles run loop sources, cannot dispatch tray click events;
/// must go through NSApplication's event queue to respond to NSStatusItem clicks and menus.
#[cfg(target_os = "macos")]
fn prepare_appkit() {
    use objc::{class, msg_send, sel, sel_impl};
    use objc::runtime::Object;

    unsafe {
        let app: *mut Object = msg_send![class!(NSApplication), sharedApplication];

        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            // NSApplicationActivationPolicyAccessory = 1 (no dock icon, but can have windows)
            let _: () = msg_send![app, setActivationPolicy: 1i64];
            let _: () = msg_send![app, finishLaunching];
        });
    }
}

#[cfg(target_os = "macos")]
fn pump_run_loop_100ms() {
    use objc::{class, msg_send, sel, sel_impl};
    use objc::runtime::Object;

    unsafe {
        let app: *mut Object = msg_send![class!(NSApplication), sharedApplication];

        prepare_appkit();

        let date_cls = class!(NSDate);
        // kCFRunLoopDefaultMode
        let mode: *mut Object = msg_send![
            class!(NSString),
            stringWithUTF8String: b"kCFRunLoopDefaultMode\0".as_ptr()
                as *const std::os::raw::c_char
        ];

        // First call waits up to 100ms; then drain remaining events (distantPast = non-blocking)
        let deadline: *mut Object =
            msg_send![date_cls, dateWithTimeIntervalSinceNow: 0.1f64];
        let past: *mut Object = msg_send![date_cls, distantPast];

        let mut first = true;
        loop {
            let date = if first { deadline } else { past };
            first = false;

            let event: *mut Object = msg_send![
                app,
                nextEventMatchingMask: u64::MAX
                untilDate: date
                inMode: mode
                dequeue: 1u8   // YES
            ];
            if event.is_null() {
                break;
            }
            let _: () = msg_send![app, sendEvent: event];
            let _: () = msg_send![app, updateWindows];
        }
    }
}

/// Linux: process glib main context so tray icon and menu can respond to clicks.
#[cfg(target_os = "linux")]
fn pump_glib_linux() {
    let ctx = glib::MainContext::default();
    while ctx.iteration(false) {}
}

/// Windows: Ctrl+C/Ctrl+Break console handler; returns 1(TRUE) to indicate handled, preventing system default process termination.
#[cfg(windows)]
unsafe extern "system" fn win32_ctrl_handler(dw_ctrl_type: u32) -> i32 {
    if dw_ctrl_type == 0 /* CTRL_C_EVENT */ || dw_ctrl_type == 1 /* CTRL_BREAK_EVENT */ {
        SIGNAL_SHUTDOWN.store(true, Ordering::SeqCst);
        1i32 // TRUE: handled, don't call next handler or ExitProcess
    } else {
        0i32 // FALSE: not handled, pass to other handlers
    }
}

/// Windows: process current thread message queue so tray icon and menu can respond to clicks.
#[cfg(target_os = "windows")]
fn pump_win32_messages() {
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        DispatchMessageW, PeekMessageW, TranslateMessage, MSG, PM_REMOVE, WM_QUIT,
    };

    let mut msg: MSG = unsafe { std::mem::zeroed() };
    while unsafe { PeekMessageW(&mut msg, 0, 0, 0, PM_REMOVE) } != 0 {
        if msg.message == WM_QUIT {
            break;
        }
        unsafe {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// stop
// ─────────────────────────────────────────────────────────────────────────────

pub async fn stop() -> Result<()> {
    let pid = match read_pid() {
        Some(p) => p,
        None => {
            println!("ℹ️  Daemon is not running (PID file not found)");
            return Ok(());
        }
    };

    if !is_running(pid) {
        println!("ℹ️  Daemon is not running (PID {} does not exist)", pid);
        remove_pid_file();
        return Ok(());
    }

    println!("⏹️  Stopping daemon (PID: {})...", pid);

    #[cfg(unix)]
    {
        unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
        for _ in 0..30 {
            std::thread::sleep(std::time::Duration::from_millis(100));
            if !is_running(pid) {
                remove_pid_file();
                println!("✅ Daemon stopped");
                return Ok(());
            }
        }
        unsafe { libc::kill(pid as libc::pid_t, libc::SIGKILL) };
        std::thread::sleep(std::time::Duration::from_millis(500));
        remove_pid_file();
        println!("✅ Daemon killed (forced)");
    }

    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
        use windows_sys::Win32::System::Threading::{OpenProcess, TerminateProcess, PROCESS_TERMINATE};
        let h = unsafe { OpenProcess(PROCESS_TERMINATE, 0, pid) };
        if h != 0 && h != -1_i32 as isize {
            unsafe {
                let _ = TerminateProcess(h as HANDLE, 0);
                CloseHandle(h as HANDLE);
            }
        }
        remove_pid_file();
        println!("✅ Daemon stopped");
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// status
// ─────────────────────────────────────────────────────────────────────────────

pub async fn status() -> Result<()> {
    let config = Config::load()?;

    match read_pid() {
        Some(pid) if is_running(pid) => {
            let uptime = get_uptime_str(pid);
            println!("Open Flow daemon status");
            println!("  Status:   ✅ Running");
            println!("  PID:      {}", pid);
            println!("  Uptime:   {}", uptime);
            println!("  Model:    {:?}", config.model_path.unwrap_or_default());
            println!("  Provider: {}", config.provider);
            println!("  Hotkey:   {}", config.hotkey);
            println!("  Trigger:  {}", config.trigger_mode);
            println!("  Log:      {}", log_path()?.display());
        }
        Some(pid) => {
            println!("Open Flow daemon status");
            println!("  Status: ❌ Not running (PID {} is stale)", pid);
            remove_pid_file();
        }
        None => {
            println!("Open Flow daemon status");
            println!("  Status: ❌ Not running");
            println!("  Start:  open-flow start");
        }
    }
    Ok(())
}

/// Get process start time via ps (display only; Windows returns N/A)
fn get_uptime_str(pid: u32) -> String {
    #[cfg(unix)]
    {
        let out = std::process::Command::new("ps")
            .args(["-p", &pid.to_string(), "-o", "etime="])
            .output();
        match out {
            Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_string(),
            _ => "unknown".to_string(),
        }
    }
    #[cfg(windows)]
    {
        let _ = pid;
        "N/A".to_string()
    }
}

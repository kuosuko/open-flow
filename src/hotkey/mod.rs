use anyhow::Result;
use std::sync::atomic::{Ordering};
use std::sync::{mpsc::Sender, Arc};
use std::thread;
use tracing::{error, info, warn};

use crate::common::types::HotkeyEvent;

/// Hotkey listener, monitors global key events via CGEventTap (macOS) or rdev (other platforms)
pub struct HotkeyListener {
    sender: Sender<HotkeyEvent>,
    hotkey: String,
}

impl HotkeyListener {
    pub fn new(sender: Sender<HotkeyEvent>, hotkey: String) -> Self {
        Self { sender, hotkey }
    }

    /// Start hotkey listener on a dedicated thread
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

    #[cfg(target_os = "macos")]
    fn run_listen_loop_macos(sender: Sender<HotkeyEvent>, hotkey: &str) -> Result<()> {
        use core_foundation::runloop::{kCFRunLoopCommonModes, CFRunLoop};
        use core_graphics::event::{
            CGEventFlags, CGEventTap, CGEventTapLocation, CGEventTapOptions, CGEventType,
            EventField, KeyCode,
        };
        use std::sync::atomic::{AtomicBool, AtomicU64};

        let is_fn_key = hotkey == "fn";
        let pressed = Arc::new(AtomicBool::new(false));
        let pressed_clone = pressed.clone();
        let press_count = Arc::new(AtomicU64::new(0));
        let release_count = Arc::new(AtomicU64::new(0));
        let pc = press_count.clone();
        let rc = release_count.clone();

        let key_name = if is_fn_key {
            "Fn"
        } else {
            "Right Command"
        };
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

                if is_fn_key {
                    // Fn key: detect via secondary Fn flag (0x800000)
                    let flags = event.get_flags();
                    let fn_down = (flags.bits() & 0x800000) != 0;

                    if fn_down {
                        let was = pressed_clone.swap(true, Ordering::SeqCst);
                        if !was {
                            let n = pc.fetch_add(1, Ordering::SeqCst) + 1;
                            info!("[Hotkey] #{} pressed (Fn)", n);
                            if let Err(e) = sender.send(HotkeyEvent::Pressed) {
                                error!("Failed to send hotkey event: {}", e);
                            }
                        }
                    } else if pressed_clone.load(Ordering::SeqCst) {
                        // Only send Released if we previously sent Pressed
                        pressed_clone.store(false, Ordering::SeqCst);
                        let n = rc.fetch_add(1, Ordering::SeqCst) + 1;
                        info!("[Hotkey] #{} released (Fn)", n);
                        if let Err(e) = sender.send(HotkeyEvent::Released) {
                            error!("Failed to send hotkey event: {}", e);
                        }
                    }
                } else {
                    // Right Command: filter by keycode first
                    let keycode =
                        event.get_integer_value_field(EventField::KEYBOARD_EVENT_KEYCODE) as u16;
                    if keycode != KeyCode::RIGHT_COMMAND {
                        return None;
                    }
                    let is_pressed = event.get_flags().contains(CGEventFlags::CGEventFlagCommand);

                    if is_pressed {
                        let was = pressed_clone.swap(true, Ordering::SeqCst);
                        if !was {
                            let n = pc.fetch_add(1, Ordering::SeqCst) + 1;
                            info!("[Hotkey] #{} pressed (Right Cmd)", n);
                            if let Err(e) = sender.send(HotkeyEvent::Pressed) {
                                error!("Failed to send hotkey event: {}", e);
                            }
                        }
                    } else {
                        let was = pressed_clone.swap(false, Ordering::SeqCst);
                        if was {
                            let n = rc.fetch_add(1, Ordering::SeqCst) + 1;
                            info!("[Hotkey] #{} released (Right Cmd)", n);
                            if let Err(e) = sender.send(HotkeyEvent::Released) {
                                error!("Failed to send hotkey event: {}", e);
                            }
                        }
                    }
                }

                None
            },
        )
        .map_err(|_| {
            anyhow::anyhow!(
                "CGEventTap creation failed, please verify Accessibility and Input Monitoring permissions are granted"
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

    #[cfg(not(target_os = "macos"))]
    fn run_listen_loop_rdev(sender: Sender<HotkeyEvent>, _hotkey: &str) -> Result<()> {
        use rdev::{listen, Event, EventType, Key};
        use std::sync::atomic::{AtomicBool, AtomicU64};

        let pressed = Arc::new(AtomicBool::new(false));
        let pressed_clone = pressed.clone();
        let press_count = Arc::new(AtomicU64::new(0));
        let release_count = Arc::new(AtomicU64::new(0));
        let pc = press_count.clone();
        let rc = release_count.clone();

        println!("⌨️  Hotkey listener started (rdev)");

        let result = listen(move |event: Event| {
            match event.event_type {
                EventType::KeyPress(Key::MetaRight) => {
                    let was = pressed_clone.swap(true, Ordering::SeqCst);
                    if !was {
                        let n = pc.fetch_add(1, Ordering::SeqCst) + 1;
                        info!("[Hotkey] event #press={} pressed", n);
                        if let Err(e) = sender.send(HotkeyEvent::Pressed) {
                            error!("Failed to send hotkey event: {}", e);
                        }
                    }
                }
                EventType::KeyRelease(Key::MetaRight) => {
                    let was = pressed_clone.swap(false, Ordering::SeqCst);
                    if was {
                        let n = rc.fetch_add(1, Ordering::SeqCst) + 1;
                        info!("[Hotkey] event #release={} released", n);
                        if let Err(e) = sender.send(HotkeyEvent::Released) {
                            error!("Failed to send hotkey event: {}", e);
                        }
                    }
                }
                _ => {}
            }
        });

        if let Err(e) = result {
            // rdev returns error when Accessibility permission is missing
            anyhow::bail!(
                "CGEventTap start failed: {:?}\n\
                 Please grant Accessibility permission: System Settings > Privacy & Security > Accessibility",
                e
            );
        }
        Ok(())
    }
}

/// Check if Accessibility permission has been granted (using macOS AXIsProcessTrusted)
pub fn check_accessibility_permission() -> bool {
    #[cfg(target_os = "macos")]
    {
        use core_foundation::base::TCFType;
        use core_foundation::dictionary::CFDictionary;
        use core_foundation::string::CFString;
        use core_foundation::boolean::CFBoolean;

        extern "C" {
            fn AXIsProcessTrustedWithOptions(options: core_foundation::dictionary::CFDictionaryRef) -> bool;
        }
        // kAXTrustedCheckOptionPrompt = "AXTrustedCheckOptionPrompt"
        let key = CFString::new("AXTrustedCheckOptionPrompt");
        let val = CFBoolean::false_value();
        let options = CFDictionary::from_CFType_pairs(&[(key.as_CFType(), val.as_CFType())]);
        unsafe { AXIsProcessTrustedWithOptions(options.as_concrete_TypeRef()) }
    }
    #[cfg(not(target_os = "macos"))]
    {
        true
    }
}

/// Check if Input Monitoring permission has been granted (required for listening to global keyboard events on macOS)
pub fn check_input_monitoring_permission() -> bool {
    #[cfg(target_os = "macos")]
    {
        #[link(name = "ApplicationServices", kind = "framework")]
        unsafe extern "C" {
            fn CGPreflightListenEventAccess() -> bool;
        }

        unsafe { CGPreflightListenEventAccess() }
    }
    #[cfg(not(target_os = "macos"))]
    {
        true
    }
}

/// Request Accessibility permission — triggers macOS system dialog
pub fn request_accessibility_permission() {
    #[cfg(target_os = "macos")]
    {
        use core_foundation::base::TCFType;
        use core_foundation::boolean::CFBoolean;
        use core_foundation::dictionary::CFDictionary;
        use core_foundation::string::CFString;

        extern "C" {
            fn AXIsProcessTrustedWithOptions(
                options: core_foundation::dictionary::CFDictionaryRef,
            ) -> bool;
        }

        // kAXTrustedCheckOptionPrompt = true → triggers the system permission dialog
        let key = CFString::new("AXTrustedCheckOptionPrompt");
        let val = CFBoolean::true_value();
        let options = CFDictionary::from_CFType_pairs(&[(key.as_CFType(), val.as_CFType())]);
        unsafe {
            AXIsProcessTrustedWithOptions(options.as_concrete_TypeRef());
        }
    }

    warn!("Accessibility permission required to listen for global hotkeys");
    println!("⚠️  Accessibility permission required");
    println!("Please go to: System Settings > Privacy & Security > Accessibility");
    println!("Add Open Flow.app to the list and enable it, then fully quit and reopen the app.");
}

/// Request Input Monitoring permission — triggers macOS system dialog
pub fn request_input_monitoring_permission() {
    #[cfg(target_os = "macos")]
    {
        #[link(name = "ApplicationServices", kind = "framework")]
        unsafe extern "C" {
            fn CGRequestListenEventAccess() -> bool;
        }

        unsafe {
            CGRequestListenEventAccess();
        }
    }

    warn!("Input Monitoring permission required to listen for global hotkeys");
    println!("\u{26a0}\u{fe0f}  Input Monitoring permission required");
    println!("Please go to: System Settings > Privacy & Security > Input Monitoring");
    println!("Add Open Flow.app to the list and enable it, then fully quit and reopen the app.");
}

/// Request microphone permission.
/// On macOS, the system dialog is triggered automatically by cpal when opening the audio device,
/// as long as NSMicrophoneUsageDescription exists in Info.plist.
/// For NotDetermined (first run), we just let cpal trigger the prompt on first recording.
pub fn request_microphone_permission() {
    info!("Microphone permission not yet granted. Dialog will appear on first recording attempt.");
    println!("   Microphone dialog will appear when you first try to record.");
    println!("   If it doesn't, go to: System Settings > Privacy & Security > Microphone");
}

/// Check microphone permission status.
/// Returns true if authorized (AVAuthorizationStatusAuthorized).
/// Returns false when undetermined (0) -- first run requires NSMicrophoneUsageDescription to trigger system dialog.
pub fn check_microphone_permission() -> bool {
    #[cfg(target_os = "macos")]
    {
        use objc::{class, msg_send, sel, sel_impl};
        use objc::runtime::Object;

        // Link AVFoundation framework (only declaration needed, no extern fn)
        #[link(name = "AVFoundation", kind = "framework")]
        extern "C" {}

        unsafe {
            // AVMediaTypeAudio = @"soun"
            let ns_string_cls = class!(NSString);
            let audio_type: *mut Object =
                msg_send![ns_string_cls, stringWithUTF8String: b"soun\0".as_ptr() as *const i8];

            // [AVCaptureDevice authorizationStatusForMediaType:] → i64
            // 0=NotDetermined 1=Restricted 2=Denied 3=Authorized
            let status: i64 =
                msg_send![class!(AVCaptureDevice), authorizationStatusForMediaType: audio_type];

            info!("Microphone TCC status: {}", status);
            status == 3
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_permission() {
        let has_permission = check_accessibility_permission();
        println!("Accessibility permission status: {}", has_permission);
    }
}

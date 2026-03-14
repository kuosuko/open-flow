pub mod chinese_convert;

use anyhow::{Context, Result};
use std::time::Duration;

/// Text injector: optional Chinese conversion + simulated typing via CGEvents (macOS)
/// or clipboard paste fallback (other platforms).
pub struct TextInjector {
    chinese_conversion: String,
}

impl TextInjector {
    pub fn new() -> Self {
        let config = crate::common::config::Config::load().unwrap_or_default();
        Self {
            chinese_conversion: config.chinese_conversion,
        }
    }

    pub async fn inject(&self, text: &str) -> Result<()> {
        // 1. Apply Chinese conversion if configured
        let text = chinese_convert::convert_chinese(text, &self.chinese_conversion);

        // 2. Also put on clipboard as backup
        if let Ok(mut clipboard) = arboard::Clipboard::new() {
            let _ = clipboard.set_text(&text);
        }

        // 3. Type text (platform-specific)
        Self::type_text(&text).await
    }

    /// macOS: Use CGEvent keyboard events to simulate real typing.
    /// This is more consistent across apps than clipboard paste (Cmd+V).
    #[cfg(target_os = "macos")]
    async fn type_text(text: &str) -> Result<()> {
        use std::ffi::c_void;

        #[link(name = "ApplicationServices", kind = "framework")]
        extern "C" {
            fn CGEventCreateKeyboardEvent(
                source: *const c_void,
                virtual_key: u16,
                key_down: bool,
            ) -> *mut c_void;
            fn CGEventKeyboardSetUnicodeString(
                event: *mut c_void,
                string_length: usize,
                unicode_string: *const u16,
            );
            fn CGEventPost(tap: u32, event: *mut c_void);
            fn CFRelease(cf: *const c_void);
        }

        fn post_unicode_chunk(chunk: &[u16]) {
            unsafe {
                // Key down
                let down = CGEventCreateKeyboardEvent(std::ptr::null(), 0, true);
                if down.is_null() {
                    return;
                }
                CGEventKeyboardSetUnicodeString(down, chunk.len(), chunk.as_ptr());
                CGEventPost(0, down); // 0 = kCGHIDEventTap
                CFRelease(down);

                // Key up
                let up = CGEventCreateKeyboardEvent(std::ptr::null(), 0, false);
                if up.is_null() {
                    return;
                }
                CGEventKeyboardSetUnicodeString(up, chunk.len(), chunk.as_ptr());
                CGEventPost(0, up);
                CFRelease(up);
            }
        }

        if text.is_empty() {
            return Ok(());
        }

        // Normalize newlines for terminal compatibility
        let normalized = text.replace('\n', "\r");
        let utf16: Vec<u16> = normalized.encode_utf16().collect();

        // Post in small chunks with a short delay between each for consistency
        const CHUNK_SIZE: usize = 20;
        const CHUNK_DELAY_MS: u64 = 5;

        for chunk in utf16.chunks(CHUNK_SIZE) {
            post_unicode_chunk(chunk);
            if CHUNK_DELAY_MS > 0 {
                tokio::time::sleep(Duration::from_millis(CHUNK_DELAY_MS)).await;
            }
        }

        Ok(())
    }

    #[cfg(target_os = "linux")]
    async fn type_text(_text: &str) -> Result<()> {
        // Linux: fall back to clipboard paste via xdotool/wtype
        let xdotool = std::process::Command::new("xdotool")
            .args(["key", "--clearmodifiers", "ctrl+v"])
            .output();

        match xdotool {
            Ok(out) if out.status.success() => return Ok(()),
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                tracing::warn!("xdotool execution failed: {}", stderr);
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                tracing::warn!("xdotool not found, trying wtype...");
            }
            Err(e) => {
                tracing::warn!("xdotool launch failed: {}", e);
            }
        }

        let wtype = std::process::Command::new("wtype")
            .args(["-M", "ctrl", "-P", "v", "-p", "v", "-m", "ctrl"])
            .output();

        match wtype {
            Ok(out) if out.status.success() => Ok(()),
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                anyhow::bail!("Both xdotool and wtype failed. wtype error: {}", stderr)
            }
            Err(_) => anyhow::bail!("Neither xdotool nor wtype found, cannot simulate paste."),
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    async fn type_text(_text: &str) -> Result<()> {
        tracing::warn!("Auto-type not supported on this platform. Text copied to clipboard — paste manually.");
        Ok(())
    }
}

impl Default for TextInjector {
    fn default() -> Self {
        Self::new()
    }
}

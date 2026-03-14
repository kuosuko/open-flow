# Open Flow Enhancements — Design Spec

**Date:** 2026-03-14
**Goal:** Add Groq API provider, Fn key / configurable hotkeys, floating indicator overlay, and native settings window to open-flow. Build everything locally, then contribute back upstream as layered PRs.

---

## 1. Provider Abstraction + Groq Backend

### Architecture

Introduce an `AsrProvider` trait to decouple the daemon from the specific speech recognition backend:

```rust
#[async_trait]
pub trait AsrProvider: Send + Sync {
    async fn transcribe(&self, audio: &[f32], sample_rate: u32) -> Result<String>;

    /// Optional warmup (e.g. load ONNX model). Default no-op.
    async fn warmup(&self) -> Result<()> { Ok(()) }

    /// Check provider readiness. Default returns Ok.
    fn check_status(&self) -> Result<String> { Ok("ready".into()) }
}
```

The trait is async to accommodate `GroqAsrProvider`'s HTTP calls. `LocalAsrProvider` wraps its synchronous CPU work in `tokio::task::spawn_blocking` internally.

- **`LocalAsrProvider`** — Wraps the existing `AsrEngine` (SenseVoiceSmall ONNX). `warmup()` loads the model. `check_status()` verifies ONNX files exist. No behavior change from current flow.
- **`GroqAsrProvider`** — Sends audio to Groq's Whisper API.
  - Encodes f32 audio buffer to WAV in-memory using `hound`.
  - Sends multipart POST to `https://api.groq.com/openai/v1/audio/transcriptions` via `reqwest` (requires adding `multipart` feature to reqwest in `Cargo.toml`).
  - Supports model selection (`whisper-large-v3-turbo` default, `whisper-large-v3` alternative).
  - `warmup()` is a no-op. `check_status()` returns error if API key is not configured.
  - Timeout: 30 seconds (matching local provider). Logs clear error on network failure.
  - Supports optional `language` parameter in config for explicit language hint (default: auto-detect).

### Config Changes (`config.toml`)

```toml
provider = "local"                      # "local" or "groq"
groq_api_key = ""                       # also accepts GROQ_API_KEY env var
groq_model = "whisper-large-v3-turbo"   # or "whisper-large-v3"
groq_language = ""                      # optional: "en", "zh", etc. Empty = auto-detect
```

API key resolution: env var `GROQ_API_KEY` takes precedence over config file value.

**Security note:** API key stored in plaintext in config file. Acceptable for a local-only tool, but users should be aware. Future improvement: macOS Keychain integration.

### Daemon Integration

At startup, the daemon constructs the appropriate `AsrProvider` based on config. The recording/transcription loop calls `provider.transcribe()` — the rest of the pipeline (hotkey detection, audio capture, text injection) is unchanged.

### Files

- `src/asr/mod.rs` — Add `AsrProvider` trait, rename `AsrEngine` to `LocalAsrProvider`.
- `src/asr/groq.rs` — New. `GroqAsrProvider` implementation.
- `src/common/config.rs` — Add `provider`, `groq_api_key`, `groq_model` fields.
- `src/daemon/mod.rs` — Use `Arc<dyn AsrProvider>` instead of `AsrEngine` directly.
- `Cargo.toml` — Add `multipart` feature to `reqwest`, add `async-trait` dependency.

---

## 2. Hotkey System Overhaul

### Configurable Hotkey

```toml
hotkey = "right_cmd"    # "right_cmd", "fn", "f13", or other key identifiers
trigger_mode = "toggle" # "toggle" or "hold"
```

### HotkeyEvent Expansion

Currently `HotkeyEvent` is a unit signal. Expand to:

```rust
pub enum HotkeyEvent {
    Pressed,
    Released,
}
```

### Trigger Modes

- **Toggle** (current default): `Pressed` → start recording. Next `Pressed` → stop + transcribe. `Released` is ignored.
- **Hold**: `Pressed` → start recording. `Released` → stop + transcribe.

The daemon checks `trigger_mode` config to interpret events.

**Hold mode and `is_processing`**: In hold mode, `Released` must always stop recording regardless of `is_processing` state — the user's physical gesture (lifting the key) is the definitive "stop" signal. The `is_processing` guard only prevents starting a *new* recording, not stopping the current one.

### Fn Key (macOS)

Reference implementation: `GroqTranscriber/src-tauri/src/platform/macos.rs` (`FnKeyListenerImpl`).

- CGEventTap on `FlagsChanged` events, detecting flag `0x800000` (secondary Fn flag).
- The Fn key CGEventTap runs on the **hotkey listener thread** (same thread as Right-Cmd tap), using that thread's CFRunLoop. It does NOT run on the main NSRunLoop thread (which is used by tray icon and overlay).
- No new permissions required — app already needs Accessibility for Right-Cmd and paste injection.

### Hotkey Abstraction

Refactor `src/hotkey/mod.rs` to select the listener based on `hotkey` config value:
- `"right_cmd"` → existing CGEventTap logic (keycode 0x36)
- `"fn"` → Fn flag detection (0x800000)
- Other keys → rdev-based listener with configurable key

### Platform Notes

- **Linux/Windows**: rdev supports press/release for standard keys. Fn key unavailable on Windows (firmware-level). Config validation should warn if Fn is selected on non-macOS.
- **Linux key-repeat suppression**: rdev on Linux generates repeated `KeyPress` events when holding a key. Must add a `was_pressed` AtomicBool guard (matching the macOS pattern) to suppress spurious events, especially important for hold mode.

### Files

- `src/hotkey/mod.rs` — Refactor into configurable listener, add Fn key detection, emit `Pressed`/`Released`.
- `src/common/types.rs` — Expand `HotkeyEvent` enum.
- `src/common/config.rs` — Add `hotkey`, `trigger_mode` fields.
- `src/daemon/mod.rs` — Handle `Pressed`/`Released` based on trigger mode.

---

## 3. Floating Indicator (Native NSWindow)

### Architecture

A borderless, transparent `NSPanel` that appears near the cursor when recording starts.

### Appearance

- Apple-style pill: rounded rect, ~200x40px
- Dark translucent background using `NSVisualEffectView` with `.hudWindow` material (native macOS blur)
- No title bar, no shadow, click-through via `setIgnoresMouseEvents:YES` (non-interactive)
- `NSWindowLevel` set above all windows

### Visual States

| RecordingState | Indicator |
|---|---|
| Idle | Hidden |
| Recording | Red pulsing dot + "Recording..." |
| Processing | Orange dot + "Transcribing..." |

### Cursor-Relative Positioning

- When transitioning to `Recording`, query `NSEvent::mouseLocation()` for current cursor position.
- Place pill ~20px below cursor, centered horizontally.
- Clamp position to the screen rect of the display containing the cursor (handles multi-monitor and mixed-DPI setups via `NSScreen::screens()` frame checks).
- Position stays fixed for the duration of the session (does not follow cursor).
- Repositioned fresh each time a new recording starts.

### Animation

- Red dot: pulsing opacity animation (0.4 → 1.0, ~1s cycle) using `CABasicAnimation` on the layer (runs on compositor thread, independent of run loop tick rate, smoother than NSTimer-based approach).

### Implementation

- New module `src/overlay/mod.rs`.
- Uses `cocoa` and `objc` crates (already dependencies).
- Runs on the main NSRunLoop thread (same as tray icon).
- Listens to the same `RecordingState` channel the tray icon uses.

### Platform Fallback

- **Linux/Windows**: Stub (no overlay). CLI output only. Same pattern as tray icon.

### Files

- `src/overlay/mod.rs` — New. NSPanel creation, state rendering, animation, cursor positioning.
- `src/lib.rs` — Add `overlay` module.
- `src/daemon/mod.rs` — Send state updates to overlay (same channel as tray).

---

## 4. Settings Window

### Architecture

A native `NSWindow` opened from a "Preferences..." menu item in the tray icon menu.

### Controls

| Setting | Control | Details |
|---|---|---|
| Provider | `NSSegmentedControl` | "Local" / "Groq" |
| Groq API Key | `NSSecureTextField` | Visible only when provider = Groq |
| Groq Model | `NSPopUpButton` | `whisper-large-v3-turbo`, `whisper-large-v3` |
| Hotkey | `NSButton` + label | Shows current key. Click → "Press a key..." capture mode. Separate "Fn" preset button. |
| Trigger Mode | `NSSegmentedControl` | "Hold" / "Toggle" |

### Behavior

- Changes save immediately to `config.toml` (no save button — macOS convention).
- Provider and hotkey changes take effect immediately via a shared `Arc<RwLock<Config>>` that both the settings window and daemon reference. When settings change, the settings window writes to config file AND updates the shared config. For hotkey changes specifically, the hotkey listener must be stopped and restarted (the CGEventTap callback captures values at creation time).
- Window is a singleton — tray menu click opens or brings to front.
- **Provider validation**: Switching to Groq without an API key shows an inline warning label below the API key field. Transcription will fall back to local if Groq fails.

### Hotkey Picker Flow

1. User clicks the hotkey button.
2. Button text changes to "Press a key..." (Escape cancels and restores previous hotkey).
3. Next key event (other than Escape) captured and set as new hotkey.
4. Button shows the new key name.
5. Separate "Fn" preset button sets it directly without listening.

### Implementation

- New module `src/settings_window/mod.rs`.
- Uses `cocoa`/`objc` crates for `NSWindow`, `NSTextField`, `NSSegmentedControl`, `NSButton`, `NSPopUpButton`.
- Objective-C selectors for button actions via `objc` callbacks.
- Runs on the main NSRunLoop thread.

### Platform Fallback

- **Linux/Windows**: Stub. Settings via config file only.

### Files

- `src/settings_window/mod.rs` — New. Window creation, controls, config read/write, hotkey picker.
- `src/lib.rs` — Add `settings_window` module.
- `src/tray/mod.rs` — Add "Preferences..." menu item.

---

## 5. .app Bundle Updates

The existing `scripts/build-app.sh` already creates a macOS `.app` bundle. Updates needed:

- Add `NSAccessibilityUsageDescription` to `Info.plist` (for Fn key / CGEventTap).
- Change `LSUIElement` to `true` for menu-bar-only behavior (no Dock icon). When the settings window opens, temporarily activate the app via `NSApp::activateIgnoringOtherApps:` so the window can receive focus and keyboard input, but the app stays out of the Dock.
- All new features (overlay, settings window, Groq provider) work within the existing bundle structure.

---

## 6. Upstream Contribution Strategy

Build everything locally on a feature branch. When ready, split into 4 PRs:

1. **PR1: Provider abstraction + Groq backend** — Least controversial, high value.
2. **PR2: Hotkey system overhaul** — Fn key, configurable hotkeys, hold/toggle modes.
3. **PR3: Floating indicator overlay** — Native NSPanel, cursor-relative.
4. **PR4: Settings window** — Native macOS preferences. Can be dropped if upstream pushes back on GUI.

Each PR should be independently functional and testable.

//! Automated hotkey test: simulate pressing Right Command, cycle "start recording -> stop transcription -> press again" and print logs.
//! Requires `RUST_LOG=info open-flow start` running in another terminal; this command only simulates key presses.

use anyhow::Result;
use std::time::{Duration, Instant};

/// Simulate a single "press and release" of Right Command
fn simulate_right_command() -> Result<()> {
    use rdev::{simulate, EventType, Key};
    let delay = Duration::from_millis(25);
    simulate(&EventType::KeyPress(Key::MetaRight))
        .map_err(|e| anyhow::anyhow!("Simulate key press failed: {:?}", e))?;
    std::thread::sleep(delay);
    simulate(&EventType::KeyRelease(Key::MetaRight))
        .map_err(|e| anyhow::anyhow!("Simulate key release failed: {:?}", e))?;
    std::thread::sleep(delay);
    Ok(())
}

/// Run hotkey simulation loop: wait for daemon to be ready, then cycle "press(start) -> wait(record) -> press(stop) -> wait(transcribe)"
pub async fn run_test_hotkey(
    cycles: u32,
    record_secs: u64,
    transcribe_wait_secs: u64,
    ready_wait_secs: u64,
) -> Result<()> {
    println!("⌨️  Hotkey Automation Test (simulating Right Command)");
    println!("   Please run in another terminal first: RUST_LOG=info open-flow start");
    println!();
    println!("   Params: {} cycles, ~{}s recording per cycle, {}s transcription wait", cycles, record_secs, transcribe_wait_secs);
    println!("   Waiting {}s after start before simulating (daemon readiness time)", ready_wait_secs);
    println!();

    std::thread::sleep(Duration::from_secs(ready_wait_secs));

    for i in 1..=cycles {
        let t0 = Instant::now();
        println!("[TestHotkey] Cycle {} — simulating keypress: start recording", i);
        simulate_right_command()?;
        std::thread::sleep(Duration::from_secs(record_secs));

        println!("[TestHotkey] Cycle {} — simulating keypress: stop and transcribe (recorded ~{}s)", i, record_secs);
        simulate_right_command()?;
        std::thread::sleep(Duration::from_secs(transcribe_wait_secs));

        let elapsed = t0.elapsed().as_secs();
        println!("[TestHotkey] Cycle {} done (elapsed {}s), next cycle...", i, elapsed);
        println!();
    }

    println!("[TestHotkey] All {} cycles complete. Check the [Hotkey] logs in the open-flow start terminal to verify behavior.", cycles);
    Ok(())
}

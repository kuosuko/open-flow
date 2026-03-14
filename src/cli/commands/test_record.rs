use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait};
use std::time::Duration;

use crate::audio::AudioCapture;

/// Test recording functionality
pub async fn test_record(duration_secs: u64) -> Result<()> {
    println!("🎙️  Test Recording");
    println!();

    // Show available devices
    let host = cpal::default_host();
    println!("Available audio input devices:");
    println!("{}", "=".repeat(50));
    if let Ok(devices) = host.input_devices() {
        for (idx, device) in devices.enumerate() {
            if let Ok(name) = device.name() {
                let is_default = host
                    .default_input_device()
                    .map(|d| d.name().ok() == Some(name.clone()))
                    .unwrap_or(false);
                println!("{} [{}] {}", if is_default { "*" } else { " " }, idx, name);
                if let Ok(cfg) = device.default_input_config() {
                    println!("    Sample rate: {}Hz, Channels: {}, Format: {:?}",
                        cfg.sample_rate().0, cfg.channels(), cfg.sample_format());
                }
            }
        }
    }
    println!("{}", "=".repeat(50));
    println!("* = default device");
    println!();

    // Initialize audio capture
    let audio_capture = AudioCapture::new()?;
    let info = audio_capture.get_info();
    println!("Audio device configuration:");
    println!("  Sample rate: {}Hz", info.sample_rate);
    println!("  Channels: {}", info.channels);
    println!("  Format: {}", info.sample_format);
    println!();

    // Prepare output path
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join("open-flow-test-recording.wav");
    
    println!("🔴 Ready to record for {} seconds...", duration_secs);
    println!("   Please get ready to speak");
    println!();

    // 3-second countdown
    for i in (1..=3).rev() {
        print!("\r   Recording countdown: {}...", i);
        std::io::Write::flush(&mut std::io::stdout())?;
        std::thread::sleep(Duration::from_secs(1));
    }
    println!("\r   Go!                        ");

    // Record directly to file
    match audio_capture.record_to_file(
        Duration::from_secs(duration_secs),
        &output_path,
    ) {
        Ok(_) => {
            println!();
            println!("✅ Test complete!");
            println!("   Recording file: {:?}", output_path);

            // Check file
            if output_path.exists() {
                let metadata = std::fs::metadata(&output_path)?;
                println!("   File size: {} bytes ({:.2} MB)",
                    metadata.len(),
                    metadata.len() as f64 / 1024.0 / 1024.0
                );
                
                // Show playback command
                println!();
                println!("📢 Play recording:");
                println!("   open {:?}", output_path);
            }
        }
        Err(e) => {
            println!();
            println!("❌ Recording failed: {}", e);
            return Err(e);
        }
    }
    
    Ok(())
}

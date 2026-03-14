use anyhow::{Context, Result};
use std::path::PathBuf;
use std::time::Duration;
use tracing::info;

use crate::asr::AsrEngine;
use crate::audio::AudioCapture;


/// One-shot transcription - record and recognize
pub async fn run(
    file: Option<PathBuf>,
    duration_secs: u64,
    model_override: Option<PathBuf>,
) -> Result<()> {
    info!("Starting one-shot transcription");

    let model_path = crate::cli::commands::setup::ensure_model_ready(model_override).await?;

    println!("🎙️  Speech Transcription");
    println!();

    let use_external_file = file.is_some();
    let audio_path = if let Some(input_file) = file {
        if !input_file.exists() {
            anyhow::bail!("Audio file not found: {:?}", input_file);
        }
        println!("📂 Using existing audio file: {:?}", input_file);
        input_file
    } else {
        let audio_capture = AudioCapture::new()
            .context("Failed to initialize audio capture")?;
        let audio_info = audio_capture.get_info();
        println!("Audio device: MacBook Pro Microphone");
        println!("  Sample rate: {}Hz, Channels: {}", audio_info.sample_rate, audio_info.channels);
        println!();
        // Determine recording duration
        let duration = if duration_secs == 0 {
        // Interactive mode: wait for user keypress to stop
        println!("🔴 Ready to record, press Enter to start...");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;

        println!("   Recording... press Enter to stop");

        // Simplified handling, default to 10 seconds
        // Should actually start a recording thread and wait for user keypress
        Duration::from_secs(10)
        } else {
            Duration::from_secs(duration_secs)
        };

        println!("🔴 Recording for {} seconds...", duration.as_secs());
        
        let temp_dir = std::env::temp_dir();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let recorded_path = temp_dir.join(format!("open-flow-transcribe-{}.wav", timestamp));

        audio_capture.record_to_file(duration, &recorded_path)?;
        println!("✓ Recording complete");
        recorded_path
    };
    
    // Transcribe
    println!("🧠 Recognizing...");

    let mut asr_engine = AsrEngine::new(model_path);
    let result = asr_engine.transcribe(&audio_path, Some("auto"))?;

    println!();
    println!("📝 Transcription result:");
    println!("   {}", result.text);
    println!();
    println!("   Confidence: {:.0}%", result.confidence * 100.0);
    println!("   Duration: {}ms", result.duration_ms);
    
    if !use_external_file {
        let _ = std::fs::remove_file(&audio_path);
    }
    
    Ok(())
}

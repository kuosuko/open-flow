//! Model preset switching: quantized (default) | fp16

use anyhow::Result;
use crate::common::config::{Config, ModelPreset};
use crate::cli::commands::setup;

/// Switch to the specified preset and optionally trigger download
pub async fn use_preset(preset: ModelPreset, download: bool) -> Result<()> {
    let mut config = Config::load()?;
    config.set_model_preset(preset)?;
    println!("✅ Current model preset: {}", preset.as_str());

    let default_dir = setup::default_model_dir(preset)?;
    if setup::model_is_ready(&default_dir) {
        println!("   Path: {} (ready)", default_dir.display());
        setup::save_model_to_config(&default_dir)?;
        if !download {
            return Ok(());
        }
        println!("   Re-checking/downloading per --download flag...");
    } else {
        println!("   Path: {} (not ready, will auto-download)", default_dir.display());
    }

    setup::download_all(None, preset, false).await?;
    setup::save_model_to_config(&default_dir)?;
    println!("✅ Model is ready. Run: open-flow start");
    Ok(())
}

/// List current preset and available presets
pub fn list() -> Result<()> {
    let config = Config::load().unwrap_or_default();
    let current = config.effective_preset();
    let path = config.model_path.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "(default directory)".to_string());

    println!("Current model preset: {}", current.as_str());
    println!("   model_path: {}", path);
    println!();
    println!("Available presets:");
    println!("  quantized   Quantized version (~200MB), default");
    println!("  fp16        High-accuracy FP16 (~450MB), switch manually: open-flow model use fp16");
    Ok(())
}

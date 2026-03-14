use anyhow::{Context, Result};
use std::io::Write;
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

/// Model download source: Hugging Face (haixuantao quantized version, consistent with ModelScope official)
const MODEL_BASE: &str = "https://huggingface.co/haixuantao/SenseVoiceSmall-onnx/resolve/main";

/// Files to download: (remote filename, local save name, expected size description)
const MODEL_FILES: &[(&str, &str, &str)] = &[
    ("model_quant.onnx", "model.onnx", "~230 MB"),
    ("am.mvn", "am.mvn", "~11 KB"),
    ("tokens.json", "tokens.json", "~344 KB"),
    ("config.yaml", "config.yaml", "~2 KB"),
];

/// Default model installation directory
pub fn default_model_dir() -> Result<PathBuf> {
    Ok(crate::common::config::Config::data_dir()?
        .join("models")
        .join("sensevoice-small"))
}

/// Check if a usable ONNX model file exists in the directory
pub fn model_is_ready(dir: &Path) -> bool {
    if !dir.is_dir() {
        return false;
    }
    let has_model = dir.join("model.onnx").exists() || dir.join("model_quant.onnx").exists();
    let has_tokens = dir.join("tokens.json").exists();
    has_model && has_tokens
}

/// Ensure model is ready: return path if already available, otherwise auto-download and write to config.
///
/// Search order:
/// 1. `model_override` (CLI --model parameter)
/// 2. model_path saved in config.toml
/// 3. Default installation directory (if exists, sync write to config)
/// 4. None found -> auto-download to default directory and write to config
pub async fn ensure_model_ready(model_override: Option<PathBuf>) -> Result<PathBuf> {
    use crate::common::config::Config;

    // 1. Explicitly specified
    if let Some(p) = model_override {
        return Ok(p);
    }

    // 2. Saved in config (exclude third-party paths like Shandianshuo, only use open-flow's own directory)
    if let Ok(config) = Config::load() {
        if let Some(ref p) = config.model_path {
            let path_str = p.to_string_lossy();
            if !path_str.contains("Shandianshuo") && !path_str.contains("shandianshuo") {
                if model_is_ready(p) {
                    return Ok(p.clone());
                }
            }
        }
    }

    // 3. Default download directory (exists but not yet written to config)
    let default_dir = default_model_dir()?;
    if model_is_ready(&default_dir) {
        save_model_to_config(&default_dir)?;
        return Ok(default_dir);
    }

    // 4. Auto-download
    println!("🔍 Local model not found, downloading automatically (first run)...");
    println!();
    download_all(None, false).await?;
    save_model_to_config(&default_dir)?;

    Ok(default_dir)
}

/// Write model path to config.toml
fn save_model_to_config(model_path: &Path) -> Result<()> {
    use crate::common::config::Config;
    let mut config = Config::load()?;
    config.model_path = Some(model_path.to_path_buf());
    config.save()?;
    Ok(())
}

/// `open-flow setup` command entry point (manual trigger, reserved for advanced use)
pub async fn run(model_dir: Option<PathBuf>, force: bool) -> Result<()> {
    download_all(model_dir.clone(), force).await?;

    // If using default directory, auto-write to config
    if model_dir.is_none() {
        let default_dir = default_model_dir()?;
        save_model_to_config(&default_dir)?;
        println!("✅ Model path has been auto-written to config. You can now run:");
        println!("   open-flow start");
    }

    Ok(())
}

/// Execute the actual download process
async fn download_all(model_dir: Option<PathBuf>, force: bool) -> Result<()> {
    let dest_dir = match model_dir {
        Some(p) => p,
        None => default_model_dir()?,
    };

    println!("📦 Open Flow Model Download");
    println!("   Target directory: {}", dest_dir.display());
    println!();

    std::fs::create_dir_all(&dest_dir)
        .with_context(|| format!("Cannot create directory: {}", dest_dir.display()))?;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .redirect(reqwest::redirect::Policy::limited(10))
        .user_agent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
        .build()?;

    for (remote_name, local_name, size_hint) in MODEL_FILES {
        let dest_path = dest_dir.join(local_name);

        if dest_path.exists() && !force {
            println!("✅ Already exists, skipping: {} ({})", local_name, size_hint);
            continue;
        }

        println!("⬇️  Downloading: {} ({})", local_name, size_hint);

        let url = format!("{}/{}", MODEL_BASE, remote_name);
        download_file(&client, &url, &dest_path)
            .await
            .with_context(|| format!("Download failed: {} -> {}", url, dest_path.display()))?;

        println!("   ✓ Done: {}", local_name);
    }

    println!();
    println!("🎉 Model download complete!");
    println!("   Path: {}", dest_dir.display());
    println!();

    Ok(())
}

async fn download_file(client: &reqwest::Client, url: &str, dest: &Path) -> Result<()> {
    let mut resp = client
        .get(url)
        .send()
        .await
        .context("HTTP request failed")?
        .error_for_status()
        .context("Server returned error status code")?;

    let total = resp.content_length();
    let mut downloaded: u64 = 0;

    let tmp_path = dest.with_extension("tmp");
    let mut file = tokio::fs::File::create(&tmp_path)
        .await
        .with_context(|| format!("Cannot create temporary file: {}", tmp_path.display()))?;

    let stdout = std::io::stdout();

    while let Some(bytes) = resp.chunk().await.context("Stream read failed")? {
        file.write_all(&bytes).await.context("Write failed")?;
        downloaded += bytes.len() as u64;

        if let Some(total) = total {
            let pct = downloaded * 100 / total;
            let mb = downloaded as f64 / 1_048_576.0;
            let total_mb = total as f64 / 1_048_576.0;
            let mut out = stdout.lock();
            let _ = write!(out, "\r   {:.1} MB / {:.1} MB  ({}%)", mb, total_mb, pct);
            let _ = out.flush();
        }
    }

    if total.is_some() {
        println!();
    }

    file.flush().await.context("Failed to flush file")?;
    drop(file);

    tokio::fs::rename(&tmp_path, dest)
        .await
        .with_context(|| format!("Rename failed: {} -> {}", tmp_path.display(), dest.display()))?;

    Ok(())
}

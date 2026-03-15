use anyhow::Result;
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use tracing::info;

const CONFIG_FILE: &str = "config.toml";

/// Model preset: only supports quantized (default) and fp16
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelPreset {
    /// HuggingFace quantized version (~200MB), default
    #[default]
    Quantized,
    /// FP16 half-precision (~450MB), higher accuracy, requires manual switch
    Fp16,
}

impl ModelPreset {
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelPreset::Quantized => "quantized",
            ModelPreset::Fp16 => "fp16",
        }
    }
}

impl std::str::FromStr for ModelPreset {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim().to_lowercase();
        match s.as_str() {
            "quantized" | "quant" => Ok(ModelPreset::Quantized),
            "fp16" | "medium" => Ok(ModelPreset::Fp16),
            _ => Err(format!("Unknown preset: {}. Options: quantized, fp16", s)),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub model_path: Option<PathBuf>,
    /// Model preset: quantized (default) | fp16. Serialized as string in config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_preset: Option<String>,
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
    /// Convert simplified Chinese to traditional: "none", "s2t" (simplified→traditional), "t2s" (traditional→simplified)
    #[serde(default)]
    pub chinese_conversion: String,
    /// Typing chunk size (UTF-16 chars per keystroke event). Smaller = more compatible. Default: 8
    #[serde(default = "default_typing_chunk_size")]
    pub typing_chunk_size: u32,
    /// Delay between typing chunks in ms. Larger = avoids double input. Default: 10
    #[serde(default = "default_typing_delay_ms")]
    pub typing_delay_ms: u32,
}

fn default_typing_chunk_size() -> u32 {
    8
}
fn default_typing_delay_ms() -> u32 {
    50
}
fn default_provider() -> String {
    "local".into()
}
fn default_groq_model() -> String {
    "whisper-large-v3-turbo".into()
}
fn default_hotkey() -> String {
    "right_cmd".into()
}
fn default_trigger_mode() -> String {
    "toggle".into()
}

impl Config {
    /// Currently effective model preset (defaults to quantized if not set in config)
    pub fn effective_preset(&self) -> ModelPreset {
        self.model_preset
            .as_deref()
            .and_then(|s| s.trim().to_lowercase().parse().ok())
            .unwrap_or_default()
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_path: None,
            model_preset: None,
            provider: default_provider(),
            groq_api_key: String::new(),
            groq_model: default_groq_model(),
            groq_language: String::new(),
            hotkey: default_hotkey(),
            trigger_mode: default_trigger_mode(),
            chinese_conversion: String::new(),
            typing_chunk_size: default_typing_chunk_size(),
            typing_delay_ms: default_typing_delay_ms(),
        }
    }
}

impl Config {
    /// Returns Groq API key: env var GROQ_API_KEY takes precedence over config file.
    pub fn resolved_groq_api_key(&self) -> String {
        std::env::var("GROQ_API_KEY").unwrap_or_else(|_| self.groq_api_key.clone())
    }

    pub fn load() -> Result<Self> {
        let config_path = Self::config_path()?;

        if !config_path.exists() {
            info!("Config file not found, creating default config");
            let config = Config::default();
            config.save()?;
            return Ok(config);
        }

        let content = fs::read_to_string(&config_path)?;
        let mut config: Config = toml::from_str(&content)?;

        // Migration: remove third-party paths like Shandianshuo, only use open-flow's own directory
        if let Some(ref p) = config.model_path {
            let s = p.to_string_lossy();
            if s.contains("Shandianshuo") || s.contains("shandianshuo") {
                config.model_path = None;
                config.save()?;
            }
        }

        if config.model_path.as_ref().map_or(false, |p| p.as_os_str().is_empty()) {
            config.model_path = None;
        }

        Ok(config)
    }

    pub fn save(&self) -> Result<()> {
        let config_path = Self::config_path()?;
        let content = toml::to_string_pretty(self)?;
        let tmp = config_path.with_extension("toml.tmp");
        fs::write(&tmp, &content)?;
        fs::rename(&tmp, &config_path)?;
        Ok(())
    }

    pub fn config_path() -> Result<PathBuf> {
        let dirs = ProjectDirs::from("com", "openflow", "open-flow")
            .ok_or_else(|| anyhow::anyhow!("Could not determine config directory"))?;

        let config_dir = dirs.config_dir();
        fs::create_dir_all(config_dir)?;

        Ok(config_dir.join(CONFIG_FILE))
    }

    pub fn data_dir() -> Result<PathBuf> {
        let dirs = ProjectDirs::from("com", "openflow", "open-flow")
            .ok_or_else(|| anyhow::anyhow!("Could not determine data directory"))?;

        let data_dir = dirs.data_dir();
        fs::create_dir_all(data_dir)?;

        Ok(data_dir.to_path_buf())
    }

    /// Set model preset and write back to config
    pub fn set_model_preset(&mut self, preset: ModelPreset) -> Result<()> {
        self.model_preset = Some(preset.as_str().to_string());
        self.save()
    }
}

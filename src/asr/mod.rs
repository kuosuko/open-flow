pub mod decoder;
pub mod groq;
pub mod onnx_inference;
pub mod preprocess;

use anyhow::{Context, Result};
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tracing::{info, warn};

use crate::asr::decoder::CTCDecoder;
use crate::asr::onnx_inference::OnnxInference;
use crate::asr::preprocess::{AudioPreprocessor, TARGET_SAMPLE_RATE};
use crate::common::types::TranscriptionResult;

/// Trait abstracting speech recognition backends (local ONNX vs cloud API).
#[async_trait]
pub trait AsrProvider: Send + Sync {
    /// Transcribe PCM audio samples. Returns transcribed text.
    async fn transcribe(&self, audio: &[f32], sample_rate: u32) -> Result<TranscriptionResult>;

    /// Optional warmup (e.g. load model, establish connection). Default no-op.
    async fn warmup(&self) -> Result<()> {
        Ok(())
    }

    /// Check provider readiness. Returns human-readable status string on success.
    fn check_status(&self) -> Result<String> {
        Ok("ready".into())
    }

    /// Provider name for display purposes.
    fn name(&self) -> &str;
}

/// Environment variables for debugging and tuning (toggleable):
/// - OPEN_FLOW_DEBUG_ASR: print features shape, encoder_out_lens, top-k logits for first/mid/last frames, and non_blank frame count
/// - OPEN_FLOW_LFR_LEFT_PAD=0: disable LFR left padding
/// - OPEN_FLOW_SKIP_CMVN: skip CMVN
/// - OPEN_FLOW_BEST_NON_BLANK=0: use pure CTC argmax (otherwise pick best non-blank per frame, useful when blank is too dominant)
/// - OPEN_FLOW_LANG_ID, OPEN_FLOW_TEXTNORM_ID: override language/textnorm inputs
pub struct AsrEngine {
    model_path: PathBuf,
    preprocessor: Option<AudioPreprocessor>,
    inference: Option<OnnxInference>,
    decoder: Option<CTCDecoder>,
    ready: bool,
}

impl AsrEngine {
    /// SenseVoice ONNX language id: consistent with FunASR/sherpa export (lid_dict)
    /// auto=0, zh=3, en=4, yue=5, ja=6, ko=7, nospeech=8
    fn resolve_language_id(language: Option<&str>) -> i32 {
        if let Ok(v) = std::env::var("OPEN_FLOW_LANG_ID") {
            if let Ok(parsed) = v.parse::<i32>() {
                return parsed;
            }
        }
        match language.unwrap_or("auto") {
            "auto" => 0,
            "zh" | "zh-cn" | "cn" => 3,
            "en" => 4,
            "yue" => 5,
            "ja" => 6,
            "ko" => 7,
            "nospeech" => 8,
            _ => 0,
        }
    }

    /// SenseVoice ONNX textnorm id: consistent with FunASR export (textnorm_dict)
    /// 0=woitn (no inverse text normalization), 1=withitn (with inverse text normalization). Can be overridden via OPEN_FLOW_TEXTNORM_ID
    fn resolve_textnorm_id() -> i32 {
        if let Ok(v) = std::env::var("OPEN_FLOW_TEXTNORM_ID") {
            if let Ok(parsed) = v.parse::<i32>() {
                return parsed;
            }
        }
        0
    }

    /// Live dictation defaults to Chinese to avoid auto misidentifying short phrases as Korean/blank.
    /// Can still be explicitly overridden via OPEN_FLOW_LANG_ID.
    fn resolve_live_language_id() -> i32 {
        if std::env::var("OPEN_FLOW_LANG_ID").is_ok() {
            return Self::resolve_language_id(Some("auto"));
        }
        Self::resolve_language_id(Some("zh"))
    }

    /// Create a new ASR engine
    pub fn new(model_path: PathBuf) -> Self {
        info!("🧠 ASR engine initializing: {:?}", model_path);

        let mut engine = Self {
            model_path,
            preprocessor: None,
            inference: None,
            decoder: None,
            ready: false,
        };

        // Try to load the model
        if let Err(e) = engine.load_model() {
            warn!("⚠️  Model loading failed: {}", e);
            warn!("   Will use mock mode");
        }

        engine
    }

    /// Find ONNX model file (supports both model.onnx and model_quant.onnx filenames)
    fn find_model_file(model_path: &Path) -> Option<PathBuf> {
        for name in &["model.onnx", "model_quant.onnx"] {
            let p = model_path.join(name);
            if p.exists() {
                return Some(p);
            }
        }
        None
    }

    /// Load model
    fn load_model(&mut self) -> Result<()> {
        let model_file = Self::find_model_file(&self.model_path)
            .ok_or_else(|| anyhow::anyhow!("Model file not found (searched model.onnx / model_quant.onnx): {:?}", self.model_path))?;
        let tokens_file = self.model_path.join("tokens.json");

        if !tokens_file.exists() {
            anyhow::bail!("Tokens file not found: {:?}", tokens_file);
        }

        info!("🔄 Loading model components...");

        // 1. Load preprocessor
        let mut pre = AudioPreprocessor::new(TARGET_SAMPLE_RATE);
        let cmvn_file = self.model_path.join("am.mvn");
        if cmvn_file.exists() {
            pre.load_cmvn_from_file(&cmvn_file)?;
            info!("✓ CMVN loaded: {:?}", cmvn_file);
        } else {
            warn!("⚠️ am.mvn not found, recognition accuracy may decrease: {:?}", cmvn_file);
        }
        self.preprocessor = Some(pre);
        info!("✓ Preprocessor loaded");

        // 2. Load ONNX model
        self.inference = Some(OnnxInference::new(&model_file)?);
        info!("✓ ONNX model loaded");

        // 3. Load decoder
        self.decoder = Some(CTCDecoder::from_tokens_file(&tokens_file)?);
        info!("✓ CTC decoder loaded");

        self.ready = true;
        info!("🎉 ASR engine fully ready!");

        Ok(())
    }

    /// Warmup: run a full inference pass with 1 second of silence to eliminate ORT JIT first-compilation overhead.
    /// Called before daemon is ready, so the first real transcription has normal latency.
    pub fn warmup(&mut self) {
        if !self.ready {
            return;
        }
        let silence = vec![0.0f32; 16000]; // 1 second of 16kHz silence
        let preprocessor = self.preprocessor.as_ref().unwrap();
        if let Ok(features) = preprocessor.process(&silence, 16000) {
            let inference = self.inference.as_mut().unwrap();
            let _ = inference.infer(&features, 0, 0); // language=auto, textnorm=off
        }
        info!("✓ Model warmup complete");
    }

    /// Transcribe directly from in-memory PCM data, avoiding disk I/O round-trip.
    /// samples: mono f32 samples (already mixed down); sample_rate: sample rate (Hz)
    pub fn transcribe_pcm(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<TranscriptionResult> {
        let start = Instant::now();

        info!("📝 Starting transcription (in-memory PCM, {} samples, {}Hz)", samples.len(), sample_rate);

        if !self.ready {
            warn!("⚠️  Model not ready, using mock transcription");
            return Ok(self.mock_transcribe());
        }

        // 1. Preprocessing
        let preprocessor = self.preprocessor.as_ref().unwrap();
        let features = preprocessor.process(samples, sample_rate)?;
        if std::env::var("OPEN_FLOW_DEBUG_ASR").is_ok() {
            info!("[DEBUG] features shape: {:?}", features.dim());
        }
        info!("✓ Feature extraction complete: {:?}", features.dim());

        // 2. ONNX inference
        let inference = self.inference.as_mut().unwrap();
        let language_id = Self::resolve_live_language_id();
        let textnorm_id = Self::resolve_textnorm_id();
        info!(
            "ASR inference params: language_id={} textnorm_id={}",
            language_id, textnorm_id
        );
        let (logits, encoder_out_lens) = inference.infer(&features, language_id, textnorm_id)?;
        if std::env::var("OPEN_FLOW_DEBUG_ASR").is_ok() {
            info!("[DEBUG] encoder_out_lens: {:?}", encoder_out_lens);
        }
        info!("✓ Inference complete: {:?}", logits.dim());

        // 3. CTC decoding
        let decoder = self.decoder.as_ref().unwrap();
        let text = decoder.decode(&logits, std::env::var("OPEN_FLOW_DEBUG_ASR").is_ok());
        info!("✓ Decoding complete: {}", text);

        Ok(TranscriptionResult {
            text,
            confidence: 0.95,
            language: Some("zh".to_string()),
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Transcribe audio file (used by the `transcribe` CLI command, internally reuses transcribe_pcm)
    pub fn transcribe(
        &mut self,
        audio_path: &Path,
        language: Option<&str>,
    ) -> Result<TranscriptionResult> {
        info!("📝 Starting transcription (file): {:?}", audio_path);

        if !audio_path.exists() {
            anyhow::bail!("Audio file not found: {:?}", audio_path);
        }

        if !self.ready {
            warn!("⚠️  Model not ready, using mock transcription");
            return Ok(self.mock_transcribe());
        }

        let audio = self.load_audio(audio_path)?;
        info!("✓ Audio loaded: {} samples", audio.data.len());

        // language parameter only used for file transcription path; in-memory path uses "auto"
        let _ = language; // Currently fixed to "auto", parameter reserved for future extension
        self.transcribe_pcm(&audio.data, audio.sample_rate)
    }

    /// Load audio file
    fn load_audio(&self, path: &Path) -> Result<AudioData> {
        use hound::WavReader;

        let reader =
            WavReader::open(path).with_context(|| format!("Cannot open audio file: {:?}", path))?;

        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        let channels = spec.channels as usize;

        info!("Audio file info:");
        info!("  Sample rate: {}Hz", sample_rate);
        info!("  Channels: {}", channels);
        info!("  Bit depth: {} bits", spec.bits_per_sample);

        // Read samples and convert to f32
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .into_samples::<f32>()
                .filter_map(|s| s.ok())
                .collect(),
            hound::SampleFormat::Int => {
                let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .into_samples::<i32>()
                    .filter_map(|s| s.ok())
                    .map(|s| s as f32 / max_val)
                    .collect()
            }
        };

        // If multi-channel, convert to mono (average)
        let mono_samples: Vec<f32> = if channels > 1 {
            samples
                .chunks(channels)
                .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
                .collect()
        } else {
            samples
        };

        Ok(AudioData {
            data: mono_samples,
            sample_rate,
        })
    }

    /// Mock transcription (used when model is not ready)
    fn mock_transcribe(&self) -> TranscriptionResult {
        std::thread::sleep(std::time::Duration::from_millis(500));

        TranscriptionResult {
            text: "[mock] Hello, this is a test.".to_string(),
            confidence: 0.95,
            language: Some("zh".to_string()),
            duration_ms: 500,
        }
    }

    /// Check ASR engine status
    pub fn check_status(&self) -> AsrStatus {
        let model_exists = self.model_path.exists();
        let onnx_exists = Self::find_model_file(&self.model_path).is_some();
        let tokens_exists = self.model_path.join("tokens.json").exists();

        AsrStatus {
            model_path: self.model_path.clone(),
            model_exists,
            onnx_exists,
            tokens_exists,
            ready: self.ready,
        }
    }
}

/// Audio data structure
struct AudioData {
    data: Vec<f32>,
    sample_rate: u32,
}

/// ASR status
#[derive(Debug, Clone)]
pub struct AsrStatus {
    pub model_path: PathBuf,
    pub model_exists: bool,
    pub onnx_exists: bool,
    #[allow(dead_code)]
    pub tokens_exists: bool,
    pub ready: bool,
}

/// Local ASR provider wrapping the existing ONNX-based AsrEngine.
/// Uses Arc<Mutex<>> so the engine can be moved into spawn_blocking.
pub struct LocalAsrProvider {
    engine: Arc<Mutex<AsrEngine>>,
}

impl LocalAsrProvider {
    pub fn new(model_path: PathBuf) -> Self {
        Self {
            engine: Arc::new(Mutex::new(AsrEngine::new(model_path))),
        }
    }
}

#[async_trait]
impl AsrProvider for LocalAsrProvider {
    async fn transcribe(&self, audio: &[f32], sample_rate: u32) -> Result<TranscriptionResult> {
        let audio = audio.to_vec();
        let engine = self.engine.clone();
        // ONNX inference is CPU-bound — run in spawn_blocking to avoid
        // blocking the tokio runtime.
        tokio::task::spawn_blocking(move || {
            engine.lock().unwrap().transcribe_pcm(&audio, sample_rate)
        })
        .await
        .context("spawn_blocking failed")?
    }

    async fn warmup(&self) -> Result<()> {
        let engine = self.engine.clone();
        tokio::task::spawn_blocking(move || {
            engine.lock().unwrap().warmup();
        })
        .await
        .context("warmup spawn_blocking failed")?;
        Ok(())
    }

    fn check_status(&self) -> Result<String> {
        let status = self.engine.lock().unwrap().check_status();
        if status.ready {
            Ok("ready".into())
        } else {
            anyhow::bail!(
                "Model not ready: {:?} (onnx={}, model={})",
                status.model_path,
                status.onnx_exists,
                status.model_exists
            )
        }
    }

    fn name(&self) -> &str {
        "local (SenseVoice)"
    }
}

#[cfg(test)]
mod regression_tests {
    use super::AsrEngine;
    use std::path::Path;

    /// Fixed audio regression: set OPEN_FLOW_REGRESSION_MODEL to point to the SenseVoice directory, run cargo test regression_mixed_zh_en -- --ignored --nocapture
    #[test]
    #[ignore]
    fn regression_mixed_zh_en() {
        let model_path = match std::env::var("OPEN_FLOW_REGRESSION_MODEL") {
            Ok(p) => Path::new(&p).to_path_buf(),
            Err(_) => return,
        };
        let wav = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/mixed_zh_en.wav");
        if !wav.exists() {
            eprintln!("Skipping regression: testdata/mixed_zh_en.wav not found");
            return;
        }
        if AsrEngine::find_model_file(&model_path).is_none() {
            eprintln!("Skipping regression: model directory has no model.onnx / model_quant.onnx");
            return;
        }
        let mut engine = AsrEngine::new(model_path);
        let result = engine.transcribe(&wav, Some("auto")).expect("transcribe");
        assert!(!result.text.is_empty(), "Regression requires non-empty output");
    }
}

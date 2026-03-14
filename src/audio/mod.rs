use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use hound::{WavSpec, WavWriter};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing::{error, info};

/// Audio capture
pub struct AudioCapture {
    device: cpal::Device,
    config: StreamConfig,
    sample_format: SampleFormat,
    sample_rate: u32,
    channels: u16,
}

/// Recording data
pub struct RecordingData {
    pub buffer: Vec<f32>,
    pub is_recording: bool,
}

impl Default for RecordingData {
    fn default() -> Self {
        Self {
            buffer: Vec::with_capacity(44100 * 10),
            is_recording: false,
        }
    }
}

impl AudioCapture {
    /// Create a new audio capture
    pub fn new() -> Result<Self> {
        info!("Initializing audio capture...");

        let host = cpal::default_host();

        // Prefer MacBook Pro microphone
        let device = host
            .input_devices()
            .ok()
            .and_then(|mut devices| {
                devices.find(|d| {
                    d.name()
                        .map(|name| name.contains("MacBook Pro"))
                        .unwrap_or(false)
                })
            })
            // If MacBook Pro microphone not found, use default device
            .or_else(|| host.default_input_device())
            .context("No input device found. Please check that a microphone is connected and enabled in System Settings.")?;

        // Get device name
        let device_name = device.name().unwrap_or_else(|_| "Unknown".to_string());
        info!("Using audio device: {}", device_name);

        // Get default config
        let supported_config = device
            .default_input_config()
            .context("Cannot get default input config")?;

        info!("Default config: {:?}", supported_config);

        let sample_format = supported_config.sample_format();
        let config: StreamConfig = supported_config.config().into();
        let sample_rate = config.sample_rate.0;
        let channels = config.channels as u16;

        info!(
            "🎙️  Audio capture initialized: {:?} @ {}Hz, {} channels",
            sample_format, sample_rate, channels
        );

        Ok(Self {
            device,
            config,
            sample_format,
            sample_rate,
            channels,
        })
    }

    /// Record audio and save directly (simplified version)
    pub fn record_to_file(&self, duration: Duration, output_path: &Path) -> Result<()> {
        info!("🔴 Recording for {} seconds...", duration.as_secs());

        // Create shared buffer
        let recording_data = Arc::new(Mutex::new(RecordingData {
            buffer: Vec::with_capacity(
                self.sample_rate as usize * duration.as_secs() as usize * self.channels as usize,
            ),
            is_recording: true,
        }));

        let recording_data_clone = recording_data.clone();

        // Error callback
        let err_fn = move |err| {
            error!("❌ Audio capture error: {}", err);
        };

        info!("Creating audio stream...");

        // Create stream based on sample format
        let stream = match self.sample_format {
            SampleFormat::F32 => self.device.build_input_stream(
                &self.config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    if let Ok(mut data_lock) = recording_data_clone.lock() {
                        if data_lock.is_recording {
                            data_lock.buffer.extend_from_slice(data);
                        }
                    }
                },
                err_fn,
                None,
            )?,
            SampleFormat::I16 => self.device.build_input_stream(
                &self.config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    if let Ok(mut data_lock) = recording_data_clone.lock() {
                        if data_lock.is_recording {
                            for &sample in data.iter() {
                                data_lock.buffer.push(sample as f32 / 32768.0);
                            }
                        }
                    }
                },
                err_fn,
                None,
            )?,
            SampleFormat::U16 => self.device.build_input_stream(
                &self.config,
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    if let Ok(mut data_lock) = recording_data_clone.lock() {
                        if data_lock.is_recording {
                            for &sample in data.iter() {
                                data_lock.buffer.push((sample as f32 - 32768.0) / 32768.0);
                            }
                        }
                    }
                },
                err_fn,
                None,
            )?,
            _ => anyhow::bail!("Unsupported sample format: {:?}", self.sample_format),
        };

        info!("Starting audio stream...");

        // Start recording
        stream.play().context("Failed to start audio stream")?;
        info!("✓ Recording started, please speak...");

        // Wait for recording to finish
        std::thread::sleep(duration);

        info!("Stopping audio stream...");

        // Stop recording
        drop(stream);

        // Mark recording as finished
        let mut data_lock = recording_data.lock().unwrap();
        data_lock.is_recording = false;

        info!("⏹️  Recording stopped, collected {} samples", data_lock.buffer.len());

        // Save recording
        if data_lock.buffer.is_empty() {
            anyhow::bail!("Audio buffer is empty, possibly no sound was recorded.\nPlease check:\n1. Microphone permission (System Settings > Privacy & Security > Microphone)\n2. Whether the microphone is working properly\n3. Whether the correct audio device is selected");
        }

        self.save_buffer_to_wav(&data_lock.buffer, output_path)?;

        Ok(())
    }

    /// Save buffer as WAV file
    pub fn save_buffer_to_wav(&self, buffer: &[f32], output_path: &Path) -> Result<()> {
        info!("💾 Saving recording to: {:?}", output_path);

        // Ensure directory exists
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // WAV file spec
        let spec = WavSpec {
            channels: self.channels,
            sample_rate: self.sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        let file = File::create(output_path)
            .with_context(|| format!("Cannot create file: {:?}", output_path))?;
        let writer = BufWriter::new(file);
        let mut wav_writer = WavWriter::new(writer, spec).context("Failed to create WAV writer")?;

        // Write samples
        for &sample in buffer {
            wav_writer.write_sample(sample)?;
        }

        wav_writer.finalize().context("Failed to finalize WAV file")?;

        let duration_secs = buffer.len() as f32 / self.sample_rate as f32 / self.channels as f32;
        info!(
            "✓ Recording saved: {} samples, {:.2} seconds",
            buffer.len(),
            duration_secs
        );

        Ok(())
    }

    /// Create and start a live recording stream, writing audio data to a shared buffer.
    /// The caller is responsible for dropping the returned Stream to stop recording.
    pub fn build_live_stream(
        &self,
        buffer: Arc<Mutex<Vec<f32>>>,
    ) -> Result<cpal::Stream> {
        let err_fn = |err: cpal::StreamError| {
            error!("❌ Audio capture error: {}", err);
        };

        let channels = self.channels as usize;

        let stream = match self.sample_format {
            SampleFormat::F32 => {
                let buf = buffer.clone();
                self.device.build_input_stream(
                    &self.config,
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        if let Ok(mut b) = buf.lock() {
                            if channels > 1 {
                                for chunk in data.chunks(channels) {
                                    b.push(chunk.iter().sum::<f32>() / channels as f32);
                                }
                            } else {
                                b.extend_from_slice(data);
                            }
                        }
                    },
                    err_fn,
                    None,
                )?
            }
            SampleFormat::I16 => {
                let buf = buffer.clone();
                self.device.build_input_stream(
                    &self.config,
                    move |data: &[i16], _: &cpal::InputCallbackInfo| {
                        if let Ok(mut b) = buf.lock() {
                            if channels > 1 {
                                for chunk in data.chunks(channels) {
                                    let mixed = chunk.iter().map(|&s| s as f32 / 32768.0).sum::<f32>() / channels as f32;
                                    b.push(mixed);
                                }
                            } else {
                                b.extend(data.iter().map(|&s| s as f32 / 32768.0));
                            }
                        }
                    },
                    err_fn,
                    None,
                )?
            }
            SampleFormat::U16 => {
                let buf = buffer.clone();
                self.device.build_input_stream(
                    &self.config,
                    move |data: &[u16], _: &cpal::InputCallbackInfo| {
                        if let Ok(mut b) = buf.lock() {
                            if channels > 1 {
                                for chunk in data.chunks(channels) {
                                    let mixed = chunk.iter().map(|&s| (s as f32 - 32768.0) / 32768.0).sum::<f32>() / channels as f32;
                                    b.push(mixed);
                                }
                            } else {
                                b.extend(data.iter().map(|&s| (s as f32 - 32768.0) / 32768.0));
                            }
                        }
                    },
                    err_fn,
                    None,
                )?
            }
            _ => anyhow::bail!("Unsupported sample format: {:?}", self.sample_format),
        };

        stream.play().context("Failed to start audio stream")?;
        Ok(stream)
    }

    /// Get audio configuration info
    pub fn get_info(&self) -> AudioInfo {
        AudioInfo {
            sample_rate: self.sample_rate,
            channels: self.channels,
            sample_format: format!("{:?}", self.sample_format),
        }
    }
}

/// Audio info
#[derive(Debug, Clone)]
pub struct AudioInfo {
    pub sample_rate: u32,
    pub channels: u16,
    pub sample_format: String,
}

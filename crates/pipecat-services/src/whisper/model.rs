use std::path::{Path, PathBuf};

/// Ensure a Whisper GGML model exists at the given cache directory,
/// downloading from HuggingFace if needed.
///
/// Model names: `tiny.en`, `base.en`, `small.en`, `medium.en`, `large`.
pub fn ensure_model(model_name: &str, cache_dir: &Path) -> std::io::Result<PathBuf> {
    let filename = format!("ggml-{model_name}.bin");
    let model_path = cache_dir.join(&filename);

    if model_path.exists() {
        return Ok(model_path);
    }

    std::fs::create_dir_all(cache_dir)?;

    let url = format!("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{filename}");

    eprintln!(
        "Downloading Whisper model '{model_name}' to {} ...",
        model_path.display()
    );

    let tmp_path = model_path.with_extension("bin.tmp");
    let status = std::process::Command::new("curl")
        .args(["-L", "--progress-bar", "-o"])
        .arg(&tmp_path)
        .arg(&url)
        .status()?;

    if !status.success() {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(std::io::Error::other(format!(
            "failed to download Whisper model from {url}"
        )));
    }

    std::fs::rename(&tmp_path, &model_path)?;
    eprintln!("Downloaded Whisper model to {}", model_path.display());
    Ok(model_path)
}

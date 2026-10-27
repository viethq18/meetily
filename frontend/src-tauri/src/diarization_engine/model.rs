// diarization_engine/model.rs
//
// Model artifacts for the diarization pipeline: pyannote segmentation-3.0
// (int8, ~1.5MB) and 3dspeaker eres2net embedding (~38MB), both hosted in the
// sherpa-onnx GitHub releases. Download is resumable (HTTP Range) with exact
// byte-size verification, mirroring gipformer_engine's download machinery at a
// smaller scale.

use anyhow::{anyhow, Context, Result};
use std::path::{Path, PathBuf};
use tokio_util::sync::CancellationToken;

/// The two required artifacts of the diarization model set.
#[derive(Clone, Copy, Debug)]
pub struct ArtifactSpec {
    pub sub_dir: &'static str,
    pub filename: &'static str,
    pub exact_bytes: u64,
    pub url: &'static str,
}

// Exact sizes from k2-fsa/sherpa-onnx releases (speaker-segmentation-models /
// speaker-recongition-models tags, verified 2026-10).
pub const DIARIZATION_ARTIFACTS: &[ArtifactSpec] = &[
    ArtifactSpec {
        sub_dir: "pyannote-segmentation-3.0",
        filename: "model.int8.onnx",
        exact_bytes: 1_540_506,
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
    },
    ArtifactSpec {
        sub_dir: "3dspeaker-eres2net",
        filename: "eres2net.int8.onnx",
        exact_bytes: 39_593_761,
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
    },
];

/// Returns true when every artifact is present with its exact expected size.
pub fn all_artifacts_present(models_dir: &Path) -> bool {
    DIARIZATION_ARTIFACTS
        .iter()
        .all(|spec| artifact_complete(models_dir, spec))
}

fn artifact_path(models_dir: &Path, spec: &ArtifactSpec) -> PathBuf {
    models_dir.join(spec.sub_dir).join(spec.filename)
}

fn artifact_complete(models_dir: &Path, spec: &ArtifactSpec) -> bool {
    match std::fs::metadata(artifact_path(models_dir, spec)) {
        Ok(meta) => meta.len() == spec.exact_bytes,
        Err(_) => false,
    }
}

/// Download status reported to the caller (for progress events).
pub struct DownloadProgress {
    pub artifact: &'static str,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
}

/// Download all missing artifacts into `models_dir`. Resumes partial files via
/// HTTP Range when the server supports it; verifies exact byte sizes at the end.
pub async fn download_models(
    models_dir: &Path,
    cancellation: &CancellationToken,
    progress: &(dyn Fn(DownloadProgress) + Send + Sync),
) -> Result<()> {
    std::fs::create_dir_all(models_dir)
        .with_context(|| format!("creating models dir {}", models_dir.display()))?;

    for spec in DIARIZATION_ARTIFACTS {
        if cancellation.is_cancelled() {
            return Err(anyhow!("Diarization model download cancelled"));
        }
        if artifact_complete(models_dir, spec) {
            log::info!(
                "Diarization artifact already present: {}/{}",
                spec.sub_dir,
                spec.filename
            );
            continue;
        }
        download_artifact(models_dir, spec, cancellation, progress).await?;
    }

    if !all_artifacts_present(models_dir) {
        return Err(anyhow!(
            "Diarization model download finished but artifacts are incomplete"
        ));
    }
    Ok(())
}

async fn download_artifact(
    models_dir: &Path,
    spec: &ArtifactSpec,
    cancellation: &CancellationToken,
    progress: &(dyn Fn(DownloadProgress) + Send + Sync),
) -> Result<()> {
    let dir = models_dir.join(spec.sub_dir);
    std::fs::create_dir_all(&dir)?;
    let dest = dir.join(spec.filename);
    let tmp = dir.join(format!("{}.part", spec.filename));

    let client = reqwest::Client::new();
    let mut range_start: u64 = match std::fs::metadata(&tmp) {
        Ok(meta) if meta.len() < spec.exact_bytes => {
            log::info!(
                "Resuming diarization artifact {} at byte {}",
                spec.filename,
                meta.len()
            );
            meta.len()
        }
        _ => {
            let _ = std::fs::remove_file(&tmp);
            0
        }
    };

    // The segmentation artifact ships as a .tar.bz2 archive; fetch and extract
    // the inner model file instead of downloading it raw.
    if spec.url.ends_with(".tar.bz2") {
        if dest.exists() {
            let meta = std::fs::metadata(&dest)?;
            if meta.len() == spec.exact_bytes {
                return Ok(());
            }
            let _ = std::fs::remove_file(&dest);
        }
        download_and_extract_tar_bz2(&client, spec, &dest, cancellation, progress).await?;
        return Ok(());
    }

    use tokio::io::AsyncWriteExt;
    let mut file = tokio::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(&tmp)
        .await?;

    let mut request = client.get(spec.url);
    if range_start > 0 {
        request = request.header("Range", format!("bytes={}-", range_start));
    }
    let response = request
        .send()
        .await
        .with_context(|| format!("starting download for {}", spec.url))?
        .error_for_status()
        .with_context(|| format!("HTTP error downloading {}", spec.url))?;

    // Server ignored our range: restart from scratch.
    if range_start > 0
        && response
            .headers()
            .get(reqwest::header::CONTENT_RANGE)
            .is_none()
    {
        range_start = 0;
        drop(file);
        file = tokio::fs::File::create(&tmp).await?;
    }

    let total = spec.exact_bytes;
    let mut downloaded = range_start;
    let mut response = response;
    while let Some(chunk) = response.chunk().await? {
        if cancellation.is_cancelled() {
            return Err(anyhow!("Diarization model download cancelled"));
        }
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        if downloaded > total {
            return Err(anyhow!(
                "downloaded more bytes than expected for {} (got {}, expected {})",
                spec.filename,
                downloaded,
                total
            ));
        }
        progress(DownloadProgress {
            artifact: spec.filename,
            downloaded_bytes: downloaded,
            total_bytes: total,
        });
    }

    if downloaded != total {
        return Err(anyhow!(
            "incomplete download for {}: got {} of {} bytes",
            spec.filename,
            downloaded,
            total
        ));
    }

    drop(file);
    tokio::fs::rename(&tmp, &dest).await?;
    log::info!("Downloaded diarization artifact {}", dest.display());
    Ok(())
}

async fn download_and_extract_tar_bz2(
    client: &reqwest::Client,
    spec: &ArtifactSpec,
    dest: &Path,
    cancellation: &CancellationToken,
    progress: &(dyn Fn(DownloadProgress) + Send + Sync),
) -> Result<()> {
    let mut response = client
        .get(spec.url)
        .send()
        .await
        .with_context(|| format!("starting download for {}", spec.url))?
        .error_for_status()
        .with_context(|| format!("HTTP error downloading {}", spec.url))?;

    let mut compressed = Vec::new();
    let mut downloaded = 0u64;
    // The tar.bz2 is ~1.5MB; no size knowledge, so just stream it fully.
    while let Some(chunk) = response.chunk().await? {
        if cancellation.is_cancelled() {
            return Err(anyhow!("Diarization model download cancelled"));
        }
        compressed.extend_from_slice(&chunk);
        downloaded += chunk.len() as u64;
        progress(DownloadProgress {
            artifact: spec.filename,
            downloaded_bytes: downloaded,
            total_bytes: 0, // unknown for the archive; UI treats 0 as indeterminate
        });
    }

    // Decompress bz2 and scan the uncompressed tar (512-byte headers) for the
    // single model file. Hand-rolled tar walk keeps the dependency surface
    // small; the format is fixed-width headers + padded payloads.
    let mut tar_bytes = Vec::new();
    let mut bz_reader = bzip2::read::BzDecoder::new(compressed.as_slice());
    std::io::Read::read_to_end(&mut bz_reader, &mut tar_bytes)
        .context("decompressing diarization model archive")?;

    let mut pos = 0usize;
    while pos + 512 <= tar_bytes.len() {
        let header = &tar_bytes[pos..pos + 512];
        if header.iter().all(|b| *b == 0) {
            break; // end-of-archive marker
        }
        let name = std::str::from_utf8(&header[0..100])
            .unwrap_or_default()
            .trim_end_matches('\0')
            .trim_end()
            .rsplit('/')
            .next()
            .unwrap_or_default()
            .to_string();
        // Octal size field, bytes 124..136.
        let size_str = std::str::from_utf8(&header[124..136])
            .unwrap_or_default()
            .trim_end_matches('\0')
            .trim();
        let size = usize::from_str_radix(size_str, 8).unwrap_or(0);
        pos += 512;
        let payload_end = (pos + size).min(tar_bytes.len());

        if name == spec.filename {
            let out = &tar_bytes[pos..payload_end];
            if out.len() as u64 != spec.exact_bytes {
                return Err(anyhow!(
                    "extracted {} has {} bytes, expected {}",
                    spec.filename,
                    out.len(),
                    spec.exact_bytes
                ));
            }
            std::fs::write(dest, out)?;
            log::info!("Extracted diarization artifact {}", dest.display());
            return Ok(());
        }
        pos += size.div_ceil(512) * 512;
    }
    Err(anyhow!(
        "did not find {} inside the archive at {}",
        spec.filename,
        spec.url
    ))
}

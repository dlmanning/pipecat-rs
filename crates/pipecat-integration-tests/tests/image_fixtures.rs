use std::io::Cursor;
use std::path::PathBuf;

use bytes::Bytes;
use image::{ImageFormat, ImageReader};
use pipecat_core::frame::*;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures")
}

/// Ensure the test fixture PNGs exist, creating them if missing.
fn ensure_fixtures() {
    let dir = fixtures_dir();
    std::fs::create_dir_all(&dir).unwrap();

    let rgb_path = dir.join("rgb_8x8.png");
    if !rgb_path.exists() {
        let img = image::RgbImage::from_fn(8, 8, |x, y| {
            image::Rgb([(x * 32) as u8, (y * 32) as u8, ((x + y) * 16) as u8])
        });
        img.save(&rgb_path).unwrap();
    }

    let rgba_path = dir.join("rgba_8x8.png");
    if !rgba_path.exists() {
        let img = image::RgbaImage::from_fn(8, 8, |x, y| {
            image::Rgba([
                (x * 32) as u8,
                (y * 32) as u8,
                ((x + y) * 16) as u8,
                128 + (x * 8) as u8,
            ])
        });
        img.save(&rgba_path).unwrap();
    }
}

// ---------------------------------------------------------------------------
// 5c-1: RGB PNG → raw pixels → JPEG round-trip
// ---------------------------------------------------------------------------

#[test]
fn rgb_png_to_jpeg_round_trip() {
    ensure_fixtures();

    let png_path = fixtures_dir().join("rgb_8x8.png");
    let img = ImageReader::open(&png_path).unwrap().decode().unwrap();
    let rgb = img.to_rgb8();

    assert_eq!(rgb.width(), 8);
    assert_eq!(rgb.height(), 8);

    // Create ImageRawFrame from raw pixels
    let raw_pixels = rgb.as_raw().clone();
    let _frame = ImageRawFrame {
        image: Bytes::from(raw_pixels.clone()),
        size: (8, 8),
        format: Some("RGB".to_string()),
    };

    // Encode to JPEG (same path as send_user_video in realtime.rs)
    let dynamic =
        image::DynamicImage::ImageRgb8(image::RgbImage::from_raw(8, 8, raw_pixels).unwrap());
    let mut jpeg_buf = Vec::new();
    let mut cursor = Cursor::new(&mut jpeg_buf);
    dynamic.write_to(&mut cursor, ImageFormat::Jpeg).unwrap();

    // Verify JPEG header
    assert!(jpeg_buf.len() > 2, "JPEG should not be empty");
    assert_eq!(jpeg_buf[0], 0xFF, "JPEG should start with 0xFF");
    assert_eq!(jpeg_buf[1], 0xD8, "JPEG should start with 0xFFD8");

    // Decode JPEG back and verify dimensions
    let decoded = image::load_from_memory_with_format(&jpeg_buf, ImageFormat::Jpeg).unwrap();
    assert_eq!(decoded.width(), 8);
    assert_eq!(decoded.height(), 8);
}

// ---------------------------------------------------------------------------
// 5c-2: RGBA PNG → raw pixels → JPEG round-trip
// ---------------------------------------------------------------------------

#[test]
fn rgba_png_to_jpeg_round_trip() {
    ensure_fixtures();

    let png_path = fixtures_dir().join("rgba_8x8.png");
    let img = ImageReader::open(&png_path).unwrap().decode().unwrap();
    let rgba = img.to_rgba8();

    assert_eq!(rgba.width(), 8);
    assert_eq!(rgba.height(), 8);

    // Create ImageRawFrame
    let raw_pixels = rgba.as_raw().clone();
    let _frame = ImageRawFrame {
        image: Bytes::from(raw_pixels.clone()),
        size: (8, 8),
        format: Some("RGBA".to_string()),
    };

    // Encode RGBA to JPEG (same path as send_user_video: DynamicImage::ImageRgba8)
    let dynamic =
        image::DynamicImage::ImageRgba8(image::RgbaImage::from_raw(8, 8, raw_pixels).unwrap());
    let mut jpeg_buf = Vec::new();
    let mut cursor = Cursor::new(&mut jpeg_buf);
    dynamic.write_to(&mut cursor, ImageFormat::Jpeg).unwrap();

    // Verify JPEG header
    assert!(jpeg_buf.len() > 2);
    assert_eq!(jpeg_buf[0], 0xFF);
    assert_eq!(jpeg_buf[1], 0xD8);

    // Decode and verify dimensions
    let decoded = image::load_from_memory_with_format(&jpeg_buf, ImageFormat::Jpeg).unwrap();
    assert_eq!(decoded.width(), 8);
    assert_eq!(decoded.height(), 8);
}

// ---------------------------------------------------------------------------
// 5c-3: Image frame construction from PNG file
// ---------------------------------------------------------------------------

#[test]
fn image_frame_from_png_file() {
    ensure_fixtures();

    let png_path = fixtures_dir().join("rgb_8x8.png");
    let png_bytes = std::fs::read(&png_path).unwrap();

    // Load from memory (as a transport would)
    let img = image::load_from_memory(&png_bytes).unwrap();
    let rgb = img.to_rgb8();

    let frame = ImageRawFrame {
        image: Bytes::from(rgb.as_raw().clone()),
        size: (rgb.width(), rgb.height()),
        format: Some("RGB".to_string()),
    };

    assert_eq!(frame.size, (8, 8));
    // RGB: 8 * 8 * 3 = 192 bytes
    assert_eq!(frame.image.len(), 192);
}

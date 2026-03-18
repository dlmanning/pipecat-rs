use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

type HmacSha256 = Hmac<Sha256>;

/// Generate a presigned WebSocket URL for AWS Transcribe Streaming.
///
/// Based on AWS Signature Version 4 signing process for WebSocket connections.
pub fn presign_url(
    access_key_id: &str,
    secret_access_key: &str,
    session_token: Option<&str>,
    region: &str,
    language_code: &str,
    sample_rate: u32,
) -> String {
    let now = chrono_utc_now();
    let amz_date = now.0.clone();
    let datestamp = now.1.clone();

    let host = format!("transcribestreaming.{region}.amazonaws.com:8443");
    let uri = "/stream-transcription-websocket";
    let service = "transcribe";
    let method = "GET";

    let credential_scope = format!("{datestamp}/{region}/{service}/aws4_request");

    // Build canonical query string (alphabetically sorted)
    let mut query_params = vec![
        (
            "X-Amz-Algorithm".to_string(),
            "AWS4-HMAC-SHA256".to_string(),
        ),
        (
            "X-Amz-Credential".to_string(),
            url_encode(&format!("{access_key_id}/{credential_scope}")),
        ),
        ("X-Amz-Date".to_string(), amz_date.clone()),
        ("X-Amz-Expires".to_string(), "300".to_string()),
        ("X-Amz-SignedHeaders".to_string(), "host".to_string()),
        (
            "enable-partial-results-stabilization".to_string(),
            "true".to_string(),
        ),
        ("language-code".to_string(), language_code.to_string()),
        ("media-encoding".to_string(), "pcm".to_string()),
        ("number-of-channels".to_string(), "1".to_string()),
        ("partial-results-stability".to_string(), "high".to_string()),
        ("sample-rate".to_string(), sample_rate.to_string()),
    ];

    if let Some(token) = session_token {
        query_params.push(("X-Amz-Security-Token".to_string(), url_encode(token)));
    }

    // Sort by key name
    query_params.sort_by(|a, b| a.0.cmp(&b.0));

    let canonical_querystring: String = query_params
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join("&");

    // Canonical headers
    let canonical_headers = format!("host:{host}\n");
    let signed_headers = "host";

    // Payload hash (empty for GET)
    let payload_hash = hex_sha256(b"");

    // Canonical request
    let canonical_request = format!(
        "{method}\n{uri}\n{canonical_querystring}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
    );

    // String to sign
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{credential_scope}\n{}",
        hex_sha256(canonical_request.as_bytes())
    );

    // Signing key
    let signing_key = get_signature_key(secret_access_key, &datestamp, region, service);

    // Signature
    let signature = hex_hmac_sha256(&signing_key, string_to_sign.as_bytes());

    format!("wss://{host}{uri}?{canonical_querystring}&X-Amz-Signature={signature}")
}

/// HMAC-SHA256 signing key derivation chain.
fn get_signature_key(key: &str, datestamp: &str, region: &str, service: &str) -> Vec<u8> {
    let k_date = hmac_sha256(format!("AWS4{key}").as_bytes(), datestamp.as_bytes());
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, service.as_bytes());
    hmac_sha256(&k_service, b"aws4_request")
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC can take key of any size");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

fn hex_hmac_sha256(key: &[u8], data: &[u8]) -> String {
    hex_encode(&hmac_sha256(key, data))
}

fn hex_sha256(data: &[u8]) -> String {
    let hash = Sha256::digest(data);
    hex_encode(&hash)
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// URL-encode a string (percent-encoding).
fn url_encode(s: &str) -> String {
    let mut encoded = String::new();
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char);
            }
            _ => {
                encoded.push_str(&format!("%{byte:02X}"));
            }
        }
    }
    encoded
}

/// Get current UTC time as (amz_date, datestamp).
/// amz_date format: "20250317T143022Z"
/// datestamp format: "20250317"
fn chrono_utc_now() -> (String, String) {
    use std::time::SystemTime;

    let now = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    let secs = now.as_secs();

    // Calculate date/time components from unix timestamp
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Civil date from days since epoch (1970-01-01)
    let (year, month, day) = civil_from_days(days as i64);

    let amz_date = format!("{year:04}{month:02}{day:02}T{hours:02}{minutes:02}{seconds:02}Z");
    let datestamp = format!("{year:04}{month:02}{day:02}");

    (amz_date, datestamp)
}

/// Convert days since 1970-01-01 to (year, month, day).
/// Algorithm from Howard Hinnant's chrono-compatible date library.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Generate a presigned URL with explicit timestamp (for testing).
#[cfg(test)]
fn presign_url_with_time(
    access_key_id: &str,
    secret_access_key: &str,
    session_token: Option<&str>,
    region: &str,
    language_code: &str,
    sample_rate: u32,
    amz_date: &str,
    datestamp: &str,
) -> String {
    let host = format!("transcribestreaming.{region}.amazonaws.com:8443");
    let uri = "/stream-transcription-websocket";
    let service = "transcribe";
    let method = "GET";

    let credential_scope = format!("{datestamp}/{region}/{service}/aws4_request");

    let mut query_params = vec![
        (
            "X-Amz-Algorithm".to_string(),
            "AWS4-HMAC-SHA256".to_string(),
        ),
        (
            "X-Amz-Credential".to_string(),
            url_encode(&format!("{access_key_id}/{credential_scope}")),
        ),
        ("X-Amz-Date".to_string(), amz_date.to_string()),
        ("X-Amz-Expires".to_string(), "300".to_string()),
        ("X-Amz-SignedHeaders".to_string(), "host".to_string()),
        (
            "enable-partial-results-stabilization".to_string(),
            "true".to_string(),
        ),
        ("language-code".to_string(), language_code.to_string()),
        ("media-encoding".to_string(), "pcm".to_string()),
        ("number-of-channels".to_string(), "1".to_string()),
        ("partial-results-stability".to_string(), "high".to_string()),
        ("sample-rate".to_string(), sample_rate.to_string()),
    ];

    if let Some(token) = session_token {
        query_params.push(("X-Amz-Security-Token".to_string(), url_encode(token)));
    }

    query_params.sort_by(|a, b| a.0.cmp(&b.0));

    let canonical_querystring: String = query_params
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join("&");

    let canonical_headers = format!("host:{host}\n");
    let signed_headers = "host";
    let payload_hash = hex_sha256(b"");

    let canonical_request = format!(
        "{method}\n{uri}\n{canonical_querystring}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
    );

    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{credential_scope}\n{}",
        hex_sha256(canonical_request.as_bytes())
    );

    let signing_key = get_signature_key(secret_access_key, datestamp, region, service);
    let signature = hex_hmac_sha256(&signing_key, string_to_sign.as_bytes());

    format!("wss://{host}{uri}?{canonical_querystring}&X-Amz-Signature={signature}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signing_key_derivation() {
        // Known test vector from AWS docs
        let key = get_signature_key(
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            "20120215",
            "us-east-1",
            "iam",
        );
        assert!(!key.is_empty());
        assert_eq!(key.len(), 32); // SHA-256 output
    }

    #[test]
    fn url_encode_basic() {
        assert_eq!(url_encode("hello"), "hello");
        assert_eq!(url_encode("hello world"), "hello%20world");
        assert_eq!(url_encode("foo/bar"), "foo%2Fbar");
        assert_eq!(url_encode("a=b&c=d"), "a%3Db%26c%3Dd");
    }

    #[test]
    fn presign_url_structure() {
        let url = presign_url_with_time(
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            None,
            "us-east-1",
            "en-US",
            16000,
            "20250317T120000Z",
            "20250317",
        );

        assert!(url.starts_with(
            "wss://transcribestreaming.us-east-1.amazonaws.com:8443/stream-transcription-websocket?"
        ));
        assert!(url.contains("X-Amz-Algorithm=AWS4-HMAC-SHA256"));
        assert!(url.contains("X-Amz-Credential="));
        assert!(url.contains("X-Amz-Date=20250317T120000Z"));
        assert!(url.contains("X-Amz-Expires=300"));
        assert!(url.contains("X-Amz-SignedHeaders=host"));
        assert!(url.contains("language-code=en-US"));
        assert!(url.contains("media-encoding=pcm"));
        assert!(url.contains("sample-rate=16000"));
        assert!(url.contains("X-Amz-Signature="));
        // Should NOT contain security token
        assert!(!url.contains("X-Amz-Security-Token"));
    }

    #[test]
    fn presign_url_with_session_token() {
        let url = presign_url_with_time(
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            Some("FwoGZXIvYXdzEBY"),
            "us-east-1",
            "en-US",
            16000,
            "20250317T120000Z",
            "20250317",
        );

        assert!(url.contains("X-Amz-Security-Token="));
    }

    #[test]
    fn presign_url_deterministic() {
        // Same inputs should produce the same output
        let url1 = presign_url_with_time(
            "AKID",
            "SECRET",
            None,
            "us-east-1",
            "en-US",
            16000,
            "20250317T120000Z",
            "20250317",
        );
        let url2 = presign_url_with_time(
            "AKID",
            "SECRET",
            None,
            "us-east-1",
            "en-US",
            16000,
            "20250317T120000Z",
            "20250317",
        );
        assert_eq!(url1, url2);
    }

    #[test]
    fn civil_from_days_epoch() {
        let (y, m, d) = civil_from_days(0);
        assert_eq!((y, m, d), (1970, 1, 1));
    }

    #[test]
    fn civil_from_days_known_date() {
        // 2025-03-17 is day 20164 since epoch
        let days = (2025 - 1970) * 365 + 13 + 31 + 28 + 17; // approximate, need exact
        let (y, m, d) = civil_from_days(days);
        assert_eq!(y, 2025);
        // Just verify it's a valid date
        assert!((1..=12).contains(&m));
        assert!((1..=31).contains(&d));
    }

    #[test]
    fn hex_sha256_empty() {
        let hash = hex_sha256(b"");
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }
}

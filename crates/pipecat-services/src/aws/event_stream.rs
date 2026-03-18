//! AWS Event Stream binary protocol encoder/decoder.
//!
//! Binary format:
//! ```text
//! [4 bytes: total_message_length (big-endian)]
//! [4 bytes: headers_length (big-endian)]
//! [4 bytes: prelude_crc32]
//! [headers_length bytes: headers]
//! [payload bytes]
//! [4 bytes: message_crc32]
//! ```
//!
//! Header format:
//! ```text
//! [1 byte: name_length]
//! [name_length bytes: name]
//! [1 byte: value_type (7 = string)]
//! [2 bytes: value_length (big-endian)]
//! [value_length bytes: value]
//! ```

/// Decoded event: (headers as name-value pairs, payload bytes).
pub type DecodedEvent = (Vec<(String, String)>, Vec<u8>);

/// Encode an audio chunk into an AWS event stream binary message.
pub fn encode_audio_event(audio_data: &[u8]) -> Vec<u8> {
    let headers = encode_headers(&[
        (":content-type", "application/octet-stream"),
        (":event-type", "AudioEvent"),
        (":message-type", "event"),
    ]);

    encode_message(&headers, audio_data)
}

/// Decode an AWS event stream binary message.
/// Returns (headers, payload) where headers is a list of (name, value) pairs.
pub fn decode_event(data: &[u8]) -> Option<DecodedEvent> {
    if data.len() < 16 {
        // Minimum: 4 (total_len) + 4 (headers_len) + 4 (prelude_crc) + 4 (msg_crc)
        return None;
    }

    let total_length = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let headers_length = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let prelude_crc = u32::from_be_bytes([data[8], data[9], data[10], data[11]]);

    // Verify prelude CRC
    let computed_prelude_crc = crc32fast::hash(&data[..8]);
    if prelude_crc != computed_prelude_crc {
        tracing::warn!(
            "AWS event stream: prelude CRC mismatch (got {prelude_crc}, expected {computed_prelude_crc})"
        );
        return None;
    }

    if data.len() < total_length {
        return None;
    }

    // Verify message CRC
    let msg_crc_offset = total_length - 4;
    let msg_crc = u32::from_be_bytes([
        data[msg_crc_offset],
        data[msg_crc_offset + 1],
        data[msg_crc_offset + 2],
        data[msg_crc_offset + 3],
    ]);
    let computed_msg_crc = crc32fast::hash(&data[..msg_crc_offset]);
    if msg_crc != computed_msg_crc {
        tracing::warn!(
            "AWS event stream: message CRC mismatch (got {msg_crc}, expected {computed_msg_crc})"
        );
        return None;
    }

    // Parse headers (start at offset 12, after prelude + prelude_crc)
    let headers_start = 12;
    let headers_end = headers_start + headers_length;
    let headers = decode_headers(&data[headers_start..headers_end]);

    // Payload is between headers and message CRC
    let payload = data[headers_end..msg_crc_offset].to_vec();

    Some((headers, payload))
}

/// Encode headers into binary format.
fn encode_headers(headers: &[(&str, &str)]) -> Vec<u8> {
    let mut buf = Vec::new();
    for (name, value) in headers {
        // Name length (1 byte) + name
        buf.push(name.len() as u8);
        buf.extend_from_slice(name.as_bytes());
        // Value type: 7 = string
        buf.push(7);
        // Value length (2 bytes big-endian) + value
        buf.extend_from_slice(&(value.len() as u16).to_be_bytes());
        buf.extend_from_slice(value.as_bytes());
    }
    buf
}

/// Decode headers from binary format.
fn decode_headers(mut data: &[u8]) -> Vec<(String, String)> {
    let mut headers = Vec::new();

    while !data.is_empty() {
        if data.is_empty() {
            break;
        }

        // Name length
        let name_len = data[0] as usize;
        data = &data[1..];
        if data.len() < name_len {
            break;
        }

        // Name
        let name = String::from_utf8_lossy(&data[..name_len]).to_string();
        data = &data[name_len..];

        if data.is_empty() {
            break;
        }

        // Value type (skip, we only handle type 7 = string)
        let _value_type = data[0];
        data = &data[1..];

        if data.len() < 2 {
            break;
        }

        // Value length
        let value_len = u16::from_be_bytes([data[0], data[1]]) as usize;
        data = &data[2..];

        if data.len() < value_len {
            break;
        }

        // Value
        let value = String::from_utf8_lossy(&data[..value_len]).to_string();
        data = &data[value_len..];

        headers.push((name, value));
    }

    headers
}

/// Encode a complete event stream message with CRC checksums.
fn encode_message(headers: &[u8], payload: &[u8]) -> Vec<u8> {
    // Total length = 4 (total_len) + 4 (headers_len) + 4 (prelude_crc) + headers + payload + 4 (msg_crc)
    let total_length = 4 + 4 + 4 + headers.len() + payload.len() + 4;

    let mut buf = Vec::with_capacity(total_length);

    // Prelude: total_length + headers_length
    buf.extend_from_slice(&(total_length as u32).to_be_bytes());
    buf.extend_from_slice(&(headers.len() as u32).to_be_bytes());

    // Prelude CRC
    let prelude_crc = crc32fast::hash(&buf[..8]);
    buf.extend_from_slice(&prelude_crc.to_be_bytes());

    // Headers + payload
    buf.extend_from_slice(headers);
    buf.extend_from_slice(payload);

    // Message CRC (covers everything before it)
    let msg_crc = crc32fast::hash(&buf);
    buf.extend_from_slice(&msg_crc.to_be_bytes());

    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_round_trip() {
        let audio = b"hello audio data";
        let encoded = encode_audio_event(audio);

        let (headers, payload) = decode_event(&encoded).unwrap();

        // Check headers
        let header_map: std::collections::HashMap<_, _> = headers.into_iter().collect();
        assert_eq!(
            header_map.get(":content-type").map(|s| s.as_str()),
            Some("application/octet-stream")
        );
        assert_eq!(
            header_map.get(":event-type").map(|s| s.as_str()),
            Some("AudioEvent")
        );
        assert_eq!(
            header_map.get(":message-type").map(|s| s.as_str()),
            Some("event")
        );

        // Check payload
        assert_eq!(payload, audio);
    }

    #[test]
    fn encode_decode_empty_payload() {
        let encoded = encode_audio_event(b"");
        let (headers, payload) = decode_event(&encoded).unwrap();
        assert_eq!(headers.len(), 3);
        assert!(payload.is_empty());
    }

    #[test]
    fn decode_too_short() {
        assert!(decode_event(&[0; 10]).is_none());
    }

    #[test]
    fn decode_bad_prelude_crc() {
        let mut encoded = encode_audio_event(b"test");
        // Corrupt prelude CRC (bytes 8-11)
        encoded[8] ^= 0xFF;
        assert!(decode_event(&encoded).is_none());
    }

    #[test]
    fn decode_bad_message_crc() {
        let mut encoded = encode_audio_event(b"test");
        // Corrupt message CRC (last 4 bytes)
        let len = encoded.len();
        encoded[len - 1] ^= 0xFF;
        assert!(decode_event(&encoded).is_none());
    }

    #[test]
    fn encode_headers_format() {
        let headers = encode_headers(&[("name", "value")]);
        // 1 (name_len) + 4 (name) + 1 (type) + 2 (value_len) + 5 (value) = 13
        assert_eq!(headers.len(), 13);
        assert_eq!(headers[0], 4); // name_len = "name".len()
        assert_eq!(&headers[1..5], b"name");
        assert_eq!(headers[5], 7); // type = string
        assert_eq!(u16::from_be_bytes([headers[6], headers[7]]), 5); // value_len
        assert_eq!(&headers[8..13], b"value");
    }

    #[test]
    fn message_length_consistency() {
        let audio = vec![0u8; 1024];
        let encoded = encode_audio_event(&audio);

        // Total length in first 4 bytes should match actual length
        let total_length =
            u32::from_be_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]) as usize;
        assert_eq!(total_length, encoded.len());
    }
}

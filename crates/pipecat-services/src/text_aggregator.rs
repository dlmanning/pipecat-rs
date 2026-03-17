/// Mode for text aggregation in TTS services.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextAggregationMode {
    /// Buffer text until a sentence boundary is detected.
    Sentence,
    /// Yield each token immediately.
    Token,
}

/// A completed aggregated text segment.
#[derive(Debug, Clone)]
pub struct AggregatedText {
    pub text: String,
    pub aggregated_by: String,
}

/// Aggregates streaming text tokens into complete segments.
///
/// In `Sentence` mode, buffers text and yields when a sentence boundary
/// (`.`, `!`, `?` followed by whitespace or end-of-input) is detected.
/// In `Token` mode, yields each text immediately.
#[derive(Debug)]
pub struct SimpleTextAggregator {
    mode: TextAggregationMode,
    buffer: String,
}

impl SimpleTextAggregator {
    pub fn new(mode: TextAggregationMode) -> Self {
        Self {
            mode,
            buffer: String::new(),
        }
    }

    /// Returns the aggregation mode.
    pub fn mode(&self) -> TextAggregationMode {
        self.mode
    }

    /// Process incoming text and return any complete segments.
    pub fn process(&mut self, text: &str) -> Vec<AggregatedText> {
        match self.mode {
            TextAggregationMode::Token => {
                if text.is_empty() {
                    return vec![];
                }
                vec![AggregatedText {
                    text: text.to_string(),
                    aggregated_by: "token".to_string(),
                }]
            }
            TextAggregationMode::Sentence => {
                self.buffer.push_str(text);
                self.extract_sentences()
            }
        }
    }

    /// Flush any remaining buffered text as a final segment.
    pub fn flush(&mut self) -> Option<AggregatedText> {
        let text = std::mem::take(&mut self.buffer);
        let text = text.trim().to_string();
        if text.is_empty() {
            return None;
        }
        Some(AggregatedText {
            text,
            aggregated_by: "sentence".to_string(),
        })
    }

    /// Reset the aggregator, discarding any buffered text.
    pub fn reset(&mut self) {
        self.buffer.clear();
    }

    /// Extract complete sentences from the buffer.
    fn extract_sentences(&mut self) -> Vec<AggregatedText> {
        let mut results = Vec::new();

        loop {
            // Find the next sentence-ending punctuation
            let boundary = self.find_sentence_boundary();
            match boundary {
                Some(pos) => {
                    let sentence = self.buffer[..pos].trim().to_string();
                    self.buffer = self.buffer[pos..].to_string();
                    // Trim leading whitespace from remainder
                    let trimmed_len = self.buffer.trim_start().len();
                    if trimmed_len < self.buffer.len() {
                        self.buffer = self.buffer.trim_start().to_string();
                    }
                    if !sentence.is_empty() {
                        results.push(AggregatedText {
                            text: sentence,
                            aggregated_by: "sentence".to_string(),
                        });
                    }
                }
                None => break,
            }
        }

        results
    }

    /// Find a sentence boundary: `.`, `!`, or `?` followed by whitespace or end of buffer.
    fn find_sentence_boundary(&self) -> Option<usize> {
        let bytes = self.buffer.as_bytes();
        for (i, &b) in bytes.iter().enumerate() {
            if b == b'.' || b == b'!' || b == b'?' {
                let end = i + 1;
                // Check if followed by whitespace or end of string
                if end >= bytes.len() || bytes[end].is_ascii_whitespace() {
                    return Some(end);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentence_mode_single_sentence() {
        let mut agg = SimpleTextAggregator::new(TextAggregationMode::Sentence);
        let results = agg.process("Hello world. ");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "Hello world.");
        assert_eq!(results[0].aggregated_by, "sentence");
    }

    #[test]
    fn sentence_mode_multi_sentence() {
        let mut agg = SimpleTextAggregator::new(TextAggregationMode::Sentence);
        let results = agg.process("First sentence. Second sentence! Third? ");
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].text, "First sentence.");
        assert_eq!(results[1].text, "Second sentence!");
        assert_eq!(results[2].text, "Third?");
    }

    #[test]
    fn sentence_mode_partial_then_complete() {
        let mut agg = SimpleTextAggregator::new(TextAggregationMode::Sentence);

        let r1 = agg.process("Hello wo");
        assert!(r1.is_empty());

        let r2 = agg.process("rld. How");
        assert_eq!(r2.len(), 1);
        assert_eq!(r2[0].text, "Hello world.");

        let r3 = agg.process(" are you? ");
        assert_eq!(r3.len(), 1);
        assert_eq!(r3[0].text, "How are you?");
    }

    #[test]
    fn sentence_mode_flush_remainder() {
        let mut agg = SimpleTextAggregator::new(TextAggregationMode::Sentence);
        agg.process("Hello world");

        let flushed = agg.flush();
        assert!(flushed.is_some());
        assert_eq!(flushed.unwrap().text, "Hello world");
    }

    #[test]
    fn sentence_mode_flush_empty() {
        let mut agg = SimpleTextAggregator::new(TextAggregationMode::Sentence);
        assert!(agg.flush().is_none());
    }

    #[test]
    fn sentence_mode_flush_after_complete_sentence() {
        let mut agg = SimpleTextAggregator::new(TextAggregationMode::Sentence);
        let results = agg.process("Done. ");
        assert_eq!(results.len(), 1);
        assert!(agg.flush().is_none());
    }

    #[test]
    fn token_mode_passthrough() {
        let mut agg = SimpleTextAggregator::new(TextAggregationMode::Token);
        let results = agg.process("Hello");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "Hello");
        assert_eq!(results[0].aggregated_by, "token");
    }

    #[test]
    fn token_mode_empty_input() {
        let mut agg = SimpleTextAggregator::new(TextAggregationMode::Token);
        let results = agg.process("");
        assert!(results.is_empty());
    }

    #[test]
    fn reset_clears_buffer() {
        let mut agg = SimpleTextAggregator::new(TextAggregationMode::Sentence);
        agg.process("partial");
        agg.reset();
        assert!(agg.flush().is_none());
    }

    #[test]
    fn abbreviations_not_split() {
        // "Dr.Smith" — no space after period, should NOT split
        let mut agg = SimpleTextAggregator::new(TextAggregationMode::Sentence);
        let results = agg.process("Dr.Smith is here. ");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "Dr.Smith is here.");
    }
}

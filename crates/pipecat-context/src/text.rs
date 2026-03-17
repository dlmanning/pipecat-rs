/// A text part collected during aggregation.
#[derive(Debug, Clone)]
pub struct TextPart {
    pub text: String,
    /// Whether the text already includes spaces between streamed tokens.
    /// When true, parts are concatenated directly. When false, a space is
    /// inserted between consecutive parts.
    pub includes_inter_part_spaces: bool,
}

/// Concatenate aggregated text parts, inserting spaces between parts that
/// don't already include inter-part spacing.
pub fn concatenate_aggregated_text(parts: &[TextPart]) -> String {
    let mut result = String::new();
    for (i, part) in parts.iter().enumerate() {
        if i > 0 && !part.includes_inter_part_spaces && !result.ends_with(' ') {
            result.push(' ');
        }
        result.push_str(&part.text);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_parts() {
        assert_eq!(concatenate_aggregated_text(&[]), "");
    }

    #[test]
    fn single_part() {
        let parts = vec![TextPart {
            text: "hello".into(),
            includes_inter_part_spaces: false,
        }];
        assert_eq!(concatenate_aggregated_text(&parts), "hello");
    }

    #[test]
    fn parts_without_spaces() {
        let parts = vec![
            TextPart {
                text: "hello".into(),
                includes_inter_part_spaces: false,
            },
            TextPart {
                text: "world".into(),
                includes_inter_part_spaces: false,
            },
        ];
        assert_eq!(concatenate_aggregated_text(&parts), "hello world");
    }

    #[test]
    fn parts_with_inter_part_spaces() {
        let parts = vec![
            TextPart {
                text: "hello ".into(),
                includes_inter_part_spaces: true,
            },
            TextPart {
                text: "world".into(),
                includes_inter_part_spaces: true,
            },
        ];
        assert_eq!(concatenate_aggregated_text(&parts), "hello world");
    }

    #[test]
    fn mixed_spacing() {
        let parts = vec![
            TextPart {
                text: "hello".into(),
                includes_inter_part_spaces: false,
            },
            TextPart {
                text: " world".into(),
                includes_inter_part_spaces: true,
            },
            TextPart {
                text: "!".into(),
                includes_inter_part_spaces: false,
            },
        ];
        assert_eq!(concatenate_aggregated_text(&parts), "hello world !");
    }

    #[test]
    fn no_double_spaces() {
        let parts = vec![
            TextPart {
                text: "hello ".into(),
                includes_inter_part_spaces: false,
            },
            TextPart {
                text: "world".into(),
                includes_inter_part_spaces: false,
            },
        ];
        // "hello " already ends with space, so no extra space added
        assert_eq!(concatenate_aggregated_text(&parts), "hello world");
    }
}

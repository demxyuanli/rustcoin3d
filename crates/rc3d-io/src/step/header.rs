//! STEP HEADER section parsing: FILE_DESCRIPTION, FILE_NAME, FILE_SCHEMA.

/// Parsed HEADER section content.
#[derive(Debug, Clone, Default)]
pub struct HeaderInfo {
    pub file_description: Vec<String>,
    pub file_name: FileName,
    pub file_schema: Vec<String>,
    pub extra: Vec<(String, String)>,
}

#[derive(Debug, Clone, Default)]
pub struct FileName {
    pub name: String,
    pub time_stamp: String,
    pub author: String,
    pub organization: String,
    pub preprocessor_version: String,
    pub originating_system: String,
    pub authorization: String,
}

/// Parse the HEADER section from raw input text between "HEADER;" and "ENDSEC;".
pub fn parse_header(input: &str) -> Option<(HeaderInfo, &str)> {
    let rest = input.trim_start();
    if !rest.starts_with("HEADER;") {
        return None;
    }
    let rest = &rest["HEADER;".len()..];
    let end = rest.find("ENDSEC;")?;
    let header_text = &rest[..end];
    let after = &rest[end + "ENDSEC;".len()..];

    let mut info = HeaderInfo::default();
    let mut pos = 0;
    while pos < header_text.len() {
        let remaining = &header_text[pos..];
        let trimmed = remaining.trim_start();
        if trimmed.is_empty() { break; }
        pos = header_text.len() - trimmed.len();
        let (keyword, after_kw) = match parse_keyword(&header_text[pos..]) {
            Some(v) => v,
            None => { log::warn!("[STEP header] malformed keyword at offset {}", pos); break; }
        };
        let after_trim = after_kw.trim_start();
        pos = header_text.len() - after_trim.len();

        if !after_trim.starts_with('(') { break; }
        let (args_str, after_args) = match extract_paren_content(&header_text[pos + 1..]) {
            Some(v) => v,
            None => { log::warn!("[STEP header] unmatched paren in {}", keyword); break; }
        };
        pos = header_text.len() - after_args.len();
        let after_trim2 = after_args.trim_start();
        if after_trim2.starts_with(';') {
            pos = header_text.len() - (after_trim2.len() - 1);
        }

        match keyword.as_str() {
            "FILE_DESCRIPTION" => {
                info.file_description = parse_string_list(&args_str);
            }
            "FILE_NAME" => {
                info.file_name = parse_file_name(&args_str);
            }
            "FILE_SCHEMA" => {
                info.file_schema = parse_string_list(&args_str);
            }
            _ => {
                info.extra.push((keyword, args_str.to_string()));
            }
        }
    }

    Some((info, after))
}

fn parse_keyword(input: &str) -> Option<(String, &str)> {
    let trimmed = input.trim_start();
    let end = trimmed.find(|c: char| !c.is_ascii_alphanumeric() && c != '_')?;
    Some((trimmed[..end].to_string(), &trimmed[end..]))
}

fn extract_paren_content(input: &str) -> Option<(&str, &str)> {
    let mut depth = 0i32;
    for (i, c) in input.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                if depth == 0 {
                    return Some((&input[..i], &input[i + 1..]));
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    None
}

fn parse_string_list(input: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut rest = input.trim();
    while !rest.is_empty() && (rest.starts_with('\'') || rest.starts_with('(')) {
        if rest.starts_with('\'') {
            match parse_single_quoted_string(&rest[1..]) {
                Some((s, after)) => { result.push(s); rest = after.trim_start(); }
                None => {
                    log::warn!("[STEP header] unterminated string in list '{}'...", &rest[..rest.len().min(30)]);
                    break;
                }
            }
        } else {
            // Nested list element: ('content'), ...
            match extract_paren_content(&rest[1..]) {
                Some((inner, after)) => { result.extend(parse_string_list(inner)); rest = after.trim_start(); }
                None => {
                    log::warn!("[STEP header] unmatched paren in list '{}'...", &rest[..rest.len().min(30)]);
                    break;
                }
            }
        }
        if rest.starts_with(',') {
            rest = rest[1..].trim_start();
        }
    }
    result
}

fn parse_single_quoted_string(input: &str) -> Option<(String, &str)> {
    let mut s = String::new();
    let mut chars = input.char_indices();
    loop {
        match chars.next() {
            Some((_, '\'')) => {
                if chars.clone().next().map_or(false, |(_, c)| c == '\'') {
                    chars.next();
                    s.push('\'');
                } else {
                    let pos = chars.clone().next().map(|(i, _)| i).unwrap_or(input.len());
                    return Some((s, &input[pos..]));
                }
            }
            Some((_, c)) => s.push(c),
            None => return None,
        }
    }
}

fn parse_file_name(input: &str) -> FileName {
    let parts = split_top_level_commas(input);
    let get = |i: usize| -> &str { parts.get(i).map(|s| s.as_str()).unwrap_or("") };
    FileName {
        name: strip_quotes(get(0)),
        time_stamp: strip_quotes(get(1)),
        author: strip_quotes(&strip_list_parens(get(2))),
        organization: strip_quotes(&strip_list_parens(get(3))),
        preprocessor_version: strip_quotes(get(4)),
        originating_system: strip_quotes(get(5)),
        authorization: strip_quotes(get(6)),
    }
}

fn split_top_level_commas(input: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (i, c) in input.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(input[start..i].trim().to_string());
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(input[start..].trim().to_string());
    parts
}

fn strip_quotes(s: &str) -> String {
    let t = s.trim();
    if t.starts_with('\'') && t.ends_with('\'') {
        t[1..t.len() - 1].replace("''", "'")
    } else {
        t.to_string()
    }
}

fn strip_list_parens(s: &str) -> String {
    let t = s.trim();
    if t.starts_with('(') && t.ends_with(')') {
        t[1..t.len()-1].to_string()
    } else {
        t.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_header_sections() {
        let input = "HEADER;
FILE_DESCRIPTION(('A STEP file'), '2;1');
FILE_NAME('example.stp', '2024-01-15T10:00:00', ('Author'), ('Org'),
  'Tool v1.0', 'ACIS 30.0', '');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN {{ 1 0 10303 214 3 1 1 }}'));
ENDSEC;
DATA;";
        let (header, after) = parse_header(input).unwrap();
        assert_eq!(header.file_schema.len(), 1);
        assert!(header.file_schema[0].contains("AUTOMOTIVE_DESIGN"));
        assert_eq!(header.file_description[0], "A STEP file");
        assert_eq!(header.file_name.name, "example.stp");
        assert_eq!(header.file_name.author, "Author");
        assert_eq!(header.file_name.organization, "Org");
        assert!(after.trim_start().starts_with("DATA;"));
    }

    #[test]
    fn test_parse_file_name() {
        let result = parse_file_name(
            "'example.stp', '2024-01-15T10:00:00', ('Author'), ('Org'), 'Tool v1', 'ACIS', ''"
        );
        assert_eq!(result.name, "example.stp");
        assert_eq!(result.author, "Author");
        assert_eq!(result.organization, "Org");
    }

    #[test]
    fn test_parse_empty_header() {
        let input = "HEADER;\nENDSEC;\nDATA;";
        let (header, _after) = parse_header(input).unwrap();
        assert!(header.file_description.is_empty());
        assert!(header.file_schema.is_empty());
    }
}

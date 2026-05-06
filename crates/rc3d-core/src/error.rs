use std::fmt;

/// Unified error type for the rustcoin3d engine.
///
/// All crates should use this type (or wrappers around it) instead of
/// bare `String` errors or unchecked `unwrap()`/`expect()` calls.
#[derive(Debug)]
pub enum EngineError {
    /// I/O error (file not found, permission denied, etc.).
    Io(std::io::Error),
    /// Failed to parse a file format (glTF, OBJ, STL, IV, FBX, etc.).
    Parse(String),
    /// Scene graph operation failed (node not found, cycle detected, etc.).
    Scene(String),
    /// Render operation failed (surface lost, pipeline creation, etc.).
    Render(String),
    /// Serialization error (JSON, binary format).
    Serde(String),
    /// Asset loading error (not found, corrupted, cancelled).
    Asset(String),
    /// An operation was cancelled before completion.
    Cancelled,
    /// A generic runtime error with a message.
    Other(String),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::Io(e) => write!(f, "I/O error: {}", e),
            EngineError::Parse(msg) => write!(f, "Parse error: {}", msg),
            EngineError::Scene(msg) => write!(f, "Scene error: {}", msg),
            EngineError::Render(msg) => write!(f, "Render error: {}", msg),
            EngineError::Serde(msg) => write!(f, "Serialization error: {}", msg),
            EngineError::Asset(msg) => write!(f, "Asset error: {}", msg),
            EngineError::Cancelled => write!(f, "Operation cancelled"),
            EngineError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for EngineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            EngineError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for EngineError {
    fn from(e: std::io::Error) -> Self {
        EngineError::Io(e)
    }
}

impl From<String> for EngineError {
    fn from(s: String) -> Self {
        EngineError::Other(s)
    }
}

impl From<&str> for EngineError {
    fn from(s: &str) -> Self {
        EngineError::Other(s.to_string())
    }
}

/// Convenience alias for results with our error type.
pub type EngineResult<T> = Result<T, EngineError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_io() {
        let err = EngineError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "file not found"));
        assert!(format!("{}", err).contains("I/O error"));
        assert!(format!("{}", err).contains("file not found"));
    }

    #[test]
    fn test_display_parse() {
        let err = EngineError::Parse("invalid syntax".into());
        assert!(format!("{}", err).contains("Parse error"));
        assert!(format!("{}", err).contains("invalid syntax"));
    }

    #[test]
    fn test_display_scene() {
        let err = EngineError::Scene("node not found".into());
        assert!(format!("{}", err).contains("Scene error"));
    }

    #[test]
    fn test_display_render() {
        let err = EngineError::Render("surface lost".into());
        assert!(format!("{}", err).contains("Render error"));
    }

    #[test]
    fn test_display_cancelled() {
        let err = EngineError::Cancelled;
        assert_eq!(format!("{}", err), "Operation cancelled");
    }

    #[test]
    fn test_from_io_error() {
        let io = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let err: EngineError = io.into();
        assert!(matches!(err, EngineError::Io(_)));
        assert!(format!("{}", err).contains("denied"));
    }

    #[test]
    fn test_from_string() {
        let err: EngineError = String::from("something broke").into();
        assert!(matches!(err, EngineError::Other(_)));
    }

    #[test]
    fn test_from_str() {
        let err: EngineError = "something broke".into();
        assert!(matches!(err, EngineError::Other(_)));
    }

    #[test]
    fn test_error_trait_source() {
        use std::error::Error;
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "nope");
        let err = EngineError::Io(io);
        assert!(err.source().is_some());
    }

    #[test]
    fn test_error_trait_no_source_for_parse() {
        use std::error::Error;
        let err = EngineError::Parse("oops".into());
        assert!(err.source().is_none());
    }
}

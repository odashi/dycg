use std::fmt;

/// Error type for this crate.
#[derive(Clone, Debug)]
pub enum Error {
    OutOfRange(String),
    ShapeMismatched(String),
    SizeMismatched(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

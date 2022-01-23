use std::fmt;

/// Error type for this crate.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    OutOfRange(String),
    InvalidNode(String),
    InvalidLength(String),
    InvalidShape(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

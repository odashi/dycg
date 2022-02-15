use std::fmt;

/// Error type for this crate.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    InvalidGraph(String),
    InvalidHardware(String),
    InvalidNode(String),
    InvalidLength(String),
    InvalidShape(String),
    OutOfRange(String),
    NotSupported(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

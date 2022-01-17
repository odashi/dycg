use std::fmt;

/// Error type for this crate.
#[derive(Clone, Debug)]
pub enum Error {
    OutOfRange(String),
    InvalidAddress(String),
    InvalidLength(String),
    InvalidShape(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

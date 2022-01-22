use crate::error::Error;
use crate::result::Result;
use std::fmt;

/// Maximum number of dimensions.
const MAX_LENGTH: usize = 8;

/// Shape of a value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    /// Number of dimensions of the shape.
    /// Only values in `dimensions_` and `strides_` with indices smaller than `length_` are available.
    length_: usize,

    /// Number of values for each dimension.
    dimensions_: [usize; MAX_LENGTH],

    /// Interval (number of values) to the next value along each dimension.
    strides_: [usize; MAX_LENGTH],
}

impl Shape {
    pub fn new0() -> Self {
        Shape {
            length_: 0,
            dimensions_: [0; MAX_LENGTH],
            strides_: [0; MAX_LENGTH],
        }
    }

    pub fn length(&self) -> usize {
        self.length_
    }

    fn check_index(&self, index: usize) -> Result<()> {
        (index < self.length_)
            .then(|| ())
            .ok_or(Error::OutOfRange(format!(
                "Shape index out of range: index:{} >= length:{}",
                index, self.length_
            )))
    }

    pub unsafe fn dimension_unchecked(&self, index: usize) -> usize {
        *self.dimensions_.get_unchecked(index)
    }

    pub fn dimension(&self, index: usize) -> Result<usize> {
        self.check_index(index)?;
        Ok(unsafe { self.dimension_unchecked(index) })
    }

    pub unsafe fn stride_unchecked(&self, index: usize) -> usize {
        *self.strides_.get_unchecked(index)
    }

    pub fn stride(&self, index: usize) -> Result<usize> {
        self.check_index(index)?;
        Ok(unsafe { self.stride_unchecked(index) })
    }

    pub fn num_values(&self) -> usize {
        self.dimensions_[..self.length_].iter().product()
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "({})",
            self.dimensions_[..self.length_]
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// Macro rules to generate a new Shape object.
#[macro_export]
macro_rules! make_shape {
    () => {
        crate::shape::Shape::new0()
    };
}

#[cfg(test)]
mod tests {
    use crate::shape::Shape;

    #[test]
    fn test_new0() {
        let shape = Shape::new0();
        assert_eq!(shape.length(), 0);
        assert!(shape.dimension(0).is_err());
        assert!(shape.stride(0).is_err());
        assert_eq!(shape.num_values(), 1);
        assert_eq!(format!("{}", shape), "()");
        assert_eq!(shape, make_shape![]);
    }
}

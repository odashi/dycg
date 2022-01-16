use crate::error::Error;
use crate::result::Result;
use std::fmt;

/// Maximum ndims of dimensions.
const MAX_SIZE: usize = 8;

/// Shape of a value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    ndims: usize,
    dims: [usize; MAX_SIZE],
}

impl Shape {
    pub fn new0() -> Self {
        Shape {
            ndims: 0,
            dims: [0, 0, 0, 0, 0, 0, 0, 0],
        }
    }

    pub fn num_dimensions(&self) -> usize {
        self.ndims
    }

    pub fn dimension(&self, index: usize) -> Result<usize> {
        if index >= self.ndims {
            return Err(Error::OutOfRange(format!(
                "index must be lower than {}, but got {}.",
                self.ndims, index
            )));
        }
        Ok(unsafe { self.dimension_unchecked(index) })
    }

    pub unsafe fn dimension_unchecked(&self, index: usize) -> usize {
        *self.dims.get_unchecked(index)
    }

    pub fn num_values(&self) -> usize {
        self.dims[..self.ndims].iter().product()
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "({})",
            self.dims[..self.ndims]
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

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
        assert_eq!(shape.num_dimensions(), 0);
        assert!(shape.dimension(0).is_err());
        assert_eq!(shape.num_values(), 1);
        assert_eq!(format!("{}", shape), "()");
        assert_eq!(shape, make_shape![]);
    }
}

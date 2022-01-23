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
}

impl Shape {
    /// Creates a new 0-dimensional shape.
    ///
    /// 0-dimensional shape has no dimension/stride values, and has only 1 element.
    /// This shape usually represents scalar values.
    ///
    /// # Returns
    ///
    /// A new `Shape` object.
    pub fn new0() -> Self {
        Shape {
            length_: 0,
            dimensions_: [0; MAX_LENGTH],
        }
    }

    /// Returns the length (number of dimensions) of this shape.
    ///
    /// # Returns
    ///
    /// The length of the shape.
    pub fn length(&self) -> usize {
        self.length_
    }

    /// Checks if the given index is valid or not in this shape.
    ///
    /// # Arguments
    ///
    /// * `index` - Index of the dimension.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - `index` is valid in this shape.
    /// * `Err(Error)` = `index` is invalid.
    pub fn check_index(&self, index: usize) -> Result<()> {
        (index < self.length_)
            .then(|| ())
            .ok_or(Error::OutOfRange(format!(
                "Shape index out of range: index:{} >= length:{}",
                index, self.length_
            )))
    }

    /// Checks if the shape represents a scalar or not.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Shape represents a scalar.
    /// * `Err(Error)` - Shape does not represent a scalar.
    pub fn check_is_scalar(&self) -> Result<()> {
        (self.length_ == 0)
            .then(|| ())
            .ok_or(Error::InvalidShape(format!(
                "Shape is not representing a scalar. length: {}",
                self.length_
            )))
    }

    /// Obtains the size of the specified dimension in this shape.
    ///
    /// # Arguments
    ///
    /// * `index` - Index of the dimension.
    ///
    /// # Returns
    ///
    /// The size of the `index`-th dimension.
    ///
    /// # Requirements
    ///
    /// `index` must be in `0..self.length()`.
    pub(crate) unsafe fn dimension_unchecked(&self, index: usize) -> usize {
        *self.dimensions_.get_unchecked(index)
    }

    /// Obtains the size of the specified dimension in this shape.
    ///
    /// # Arguments
    ///
    /// * `index` - Index of the dimension.
    ///
    /// # Returns
    ///
    /// * `Ok(usize)` - The size of the `index`-th dimension.
    /// * `Err(Error)` - `index` is out-of-range.
    pub fn dimension(&self, index: usize) -> Result<usize> {
        self.check_index(index)?;
        Ok(unsafe { self.dimension_unchecked(index) })
    }

    /// Calculates the number of elements represented by this shape.
    ///
    /// The number of elements is defined as usually the product of all valid dimension sizes.
    /// This value could become 0 if some dimension has lengths of 0.
    ///
    /// # Returns
    ///
    /// The number of elements represented by the shape.
    pub fn get_num_elements(&self) -> usize {
        self.dimensions_[..self.length_].iter().product()
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Dimensions == []        => "()"
        // Dimensions == [42]      => "(42)"
        // Dimensions == [1, 2, 3] => "(1, 2, 3)"
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
        assert!(shape.check_index(0).is_err());
        assert!(shape.check_is_scalar().is_ok());
        assert!(shape.dimension(0).is_err());
        assert_eq!(shape.get_num_elements(), 1);
        assert_eq!(format!("{}", shape), "()");
        assert_eq!(shape, make_shape![]);
    }
}

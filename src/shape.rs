use crate::error::Error;
use crate::result::Result;
use std::fmt;
use std::mem::size_of;

/// Maximum number of dimensions.
const MAX_NUM_DIMENSIONS: usize = 8;

/// Shape of a value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Shape {
    /// Number of dimensions of the shape.
    /// Only values in `dimensions` with indices smaller than `num_dimensions` are available.
    num_dimensions: usize,

    /// Number of values for each dimension.
    dimensions: [usize; MAX_NUM_DIMENSIONS],

    /// Number of elements in this shape.
    /// Since this value is frequently used, the value is calculated and cached at the
    /// initialization.
    num_elements: usize,
}

impl Shape {
    /// Common function to create a new `Shape` with calculating inner statistics.
    ///
    /// # Arguments
    ///
    /// * `num_dimensions` - Number of dimensions in the new shape.
    /// * `dimensions` - Number of elements for each dimension.
    ///
    /// # Returns
    ///
    /// A new `Shape` object.
    ///
    /// # Panics
    ///
    /// `num_dimensions` exceeds MAX_NUM_DIMENSIONS.
    fn new_inner(num_dimensions: usize, dimensions: [usize; MAX_NUM_DIMENSIONS]) -> Self {
        Self {
            num_dimensions,
            dimensions,
            num_elements: dimensions[..num_dimensions].iter().product(),
        }
    }

    /// Creates a new 0-dimensional shape.
    ///
    /// 0-dimensional shape has no dimension values, and has exactly 1 element.
    /// This shape usually represents scalar values.
    ///
    /// # Returns
    ///
    /// A new `Shape` object.
    pub fn new0() -> Self {
        Self::new_inner(0, [0; MAX_NUM_DIMENSIONS])
    }

    /// Creates a new 1-dimensional shape.
    ///
    /// # Arguments
    ///
    /// * `axis0` - Number of elements in the 0th axis.
    ///
    /// # Returns
    ///
    /// A new `Shape` object.
    pub fn new1(axis0: usize) -> Self {
        Self::new_inner(1, [axis0, 0, 0, 0, 0, 0, 0, 0])
    }

    /// Returns the number of dimensions of this shape.
    ///
    /// # Returns
    ///
    /// The number of dimensions of the shape.
    pub fn num_dimensions(&self) -> usize {
        self.num_dimensions
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
        (index < self.num_dimensions).then(|| ()).ok_or_else(|| {
            Error::OutOfRange(format!(
                "Shape index out of range: index:{} >= num_dimensions:{}",
                index, self.num_dimensions
            ))
        })
    }

    /// Checks if the shape represents a scalar or not.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Shape represents a scalar.
    /// * `Err(Error)` - Shape does not represent a scalar.
    pub fn check_is_scalar(&self) -> Result<()> {
        (self.num_dimensions == 0).then(|| ()).ok_or_else(|| {
            Error::InvalidShape(format!(
                "Shape is not representing a scalar. num_dimensions: {}",
                self.num_dimensions
            ))
        })
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
    /// `index` must be in `0..self.num_dimensions()`.
    pub(crate) unsafe fn dimension_unchecked(&self, index: usize) -> usize {
        *self.dimensions.get_unchecked(index)
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
    /// This value could become 0 if some dimension has num_dimensions of 0.
    ///
    /// # Returns
    ///
    /// The number of elements represented by the shape.
    pub fn num_elements(&self) -> usize {
        self.num_elements
    }

    /// Calculates the memory size required for this shape with a specific value type.
    ///
    /// # Type arguments
    ///
    /// * `T` - Value type.
    ///
    /// # Returns
    ///
    /// The size in bytes required to represent this shape.
    pub fn memory_size<T: Sized>(&self) -> usize {
        self.num_elements * size_of::<T>()
    }

    /// Obtains the resulting shape of elementwise binary operation.
    ///
    /// This function returns a shape of the result of `self (op) other` operation.
    /// Specifically, this function checks the equality of both shapes, then results a copy of
    /// `self`.
    /// This function does not evaluate broadcasting.
    ///
    /// # Arguments
    ///
    /// * `other` - Right-hand side argument.
    ///
    /// # Returns
    ///
    /// * `Ok(Self)` - The shape of the reult of elementwise binary operation.
    /// * `Err(Error)` - Operation failed.
    pub fn elementwise(&self, other: &Self) -> Result<Self> {
        if self == other {
            Ok(self.clone())
        } else {
            Err(Error::InvalidShape(format!(
                "Elementwise operation can not be evaluated for shapes {} and {}.",
                self, other
            )))
        }
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // length = 0, dimensions = [...]          => "()"
        // length = 1, dimensions = [3, ...]       => "(3)"
        // length = 2, dimensions = [1, 2, 3, ...] => "(1, 2, 3)"
        write!(
            f,
            "({})",
            self.dimensions[..self.num_dimensions]
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
    ( $axis0:expr ) => {
        crate::shape::Shape::new1($axis0)
    };
}

#[cfg(test)]
mod tests {
    use crate::shape::{Shape, MAX_NUM_DIMENSIONS};

    #[test]
    fn test_new0() {
        let shape = Shape::new0();
        assert_eq!(shape.num_dimensions, 0);
        assert_eq!(shape.dimensions, [0; MAX_NUM_DIMENSIONS]);
        assert_eq!(shape.num_elements, 1);
    }

    #[test]
    fn test_new1() {
        let shape = Shape::new1(3);
        assert_eq!(shape.num_dimensions, 1);
        assert_eq!(shape.dimensions, [3, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(shape.num_elements, 3);
    }

    #[test]
    fn test_num_dimensions() {
        assert_eq!(Shape::new0().num_dimensions(), 0);
        assert_eq!(Shape::new1(3).num_dimensions(), 1);
    }

    #[test]
    fn test_check_index() {
        assert!(Shape::new0().check_index(0).is_err());
        assert!(Shape::new1(3).check_index(0).is_ok());
        assert!(Shape::new1(3).check_index(1).is_err());
    }

    #[test]
    fn test_check_is_scalar() {
        assert!(Shape::new0().check_is_scalar().is_ok());
        assert!(Shape::new1(3).check_is_scalar().is_err());
    }

    #[test]
    fn test_dimension_unchecked() {
        unsafe {
            assert_eq!(Shape::new1(3).dimension_unchecked(0), 3);
        }
    }

    #[test]
    fn test_dimension() {
        assert!(Shape::new0().dimension(0).is_err());
        assert_eq!(Shape::new1(3).dimension(0), Ok(3));
        assert!(Shape::new1(3).dimension(1).is_err());
    }

    #[test]
    fn test_num_elements() {
        assert_eq!(Shape::new0().num_elements(), 1);
        assert_eq!(Shape::new1(0).num_elements(), 0);
        assert_eq!(Shape::new1(3).num_elements(), 3);
    }

    #[test]
    fn test_memory_size() {
        // 0-dimensional
        assert_eq!(Shape::new0().memory_size::<bool>(), 1);
        assert_eq!(Shape::new0().memory_size::<i8>(), 1);
        assert_eq!(Shape::new0().memory_size::<i16>(), 2);
        assert_eq!(Shape::new0().memory_size::<i32>(), 4);
        assert_eq!(Shape::new0().memory_size::<i64>(), 8);
        assert_eq!(Shape::new0().memory_size::<u8>(), 1);
        assert_eq!(Shape::new0().memory_size::<u16>(), 2);
        assert_eq!(Shape::new0().memory_size::<u32>(), 4);
        assert_eq!(Shape::new0().memory_size::<u64>(), 8);
        assert_eq!(Shape::new0().memory_size::<f32>(), 4);
        assert_eq!(Shape::new0().memory_size::<f64>(), 8);

        // 1-dimensional
        assert_eq!(Shape::new1(0).memory_size::<bool>(), 0);
        assert_eq!(Shape::new1(0).memory_size::<i8>(), 0);
        assert_eq!(Shape::new1(0).memory_size::<i16>(), 0);
        assert_eq!(Shape::new1(0).memory_size::<i32>(), 0);
        assert_eq!(Shape::new1(0).memory_size::<i64>(), 0);
        assert_eq!(Shape::new1(0).memory_size::<u8>(), 0);
        assert_eq!(Shape::new1(0).memory_size::<u16>(), 0);
        assert_eq!(Shape::new1(0).memory_size::<u32>(), 0);
        assert_eq!(Shape::new1(0).memory_size::<u64>(), 0);
        assert_eq!(Shape::new1(0).memory_size::<f32>(), 0);
        assert_eq!(Shape::new1(0).memory_size::<f64>(), 0);

        assert_eq!(Shape::new1(3).memory_size::<bool>(), 3);
        assert_eq!(Shape::new1(3).memory_size::<i8>(), 3);
        assert_eq!(Shape::new1(3).memory_size::<i16>(), 6);
        assert_eq!(Shape::new1(3).memory_size::<i32>(), 12);
        assert_eq!(Shape::new1(3).memory_size::<i64>(), 24);
        assert_eq!(Shape::new1(3).memory_size::<u8>(), 3);
        assert_eq!(Shape::new1(3).memory_size::<u16>(), 6);
        assert_eq!(Shape::new1(3).memory_size::<u32>(), 12);
        assert_eq!(Shape::new1(3).memory_size::<u64>(), 24);
        assert_eq!(Shape::new1(3).memory_size::<f32>(), 12);
        assert_eq!(Shape::new1(3).memory_size::<f64>(), 24);
    }

    #[test]
    fn test_elementwise() {
        assert_eq!(Shape::new0().elementwise(&Shape::new0()), Ok(Shape::new0()));
        assert!(Shape::new0().elementwise(&Shape::new1(3)).is_err());
        assert!(Shape::new1(3).elementwise(&Shape::new0()).is_err());
        assert!(Shape::new1(3).elementwise(&Shape::new1(0)).is_err());
        assert!(Shape::new1(3).elementwise(&Shape::new1(1)).is_err());
        assert_eq!(
            Shape::new1(3).elementwise(&Shape::new1(3)),
            Ok(Shape::new1(3))
        );
    }

    #[test]
    fn test_fmt() {
        assert_eq!(format!("{}", Shape::new0()), "()");
        assert_eq!(format!("{}", Shape::new1(0)), "(0)");
        assert_eq!(format!("{}", Shape::new1(3)), "(3)");
    }

    #[test]
    fn test_make_shape() {
        assert_eq!(make_shape![], Shape::new0());
        assert_eq!(make_shape![0], Shape::new1(0));
        assert_eq!(make_shape![3], Shape::new1(3));
    }
}

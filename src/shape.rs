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
    /// Creates a new n-dimensional shape.
    ///
    /// # Arguments
    ///
    /// * `dimensions` - Number of elements for each axis.
    ///
    /// # Returns
    ///
    /// A new `Shape` object.
    ///
    /// # Panics
    ///
    /// The const parameter `N` is larger than `MAX_NUM_DIMENSIONS`.
    pub fn new<const N: usize>(dimensions: [usize; N]) -> Self {
        let (dimensions, num_elements) = {
            let mut buffer = [0usize; MAX_NUM_DIMENSIONS];
            let mut size = 1usize;
            for i in 0..N {
                let d = dimensions[i];
                buffer[i] = d;
                size *= d;
            }
            (buffer, size)
        };
        Self {
            num_dimensions: N,
            dimensions,
            num_elements,
        }
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

#[cfg(test)]
mod tests {
    use crate::shape::Shape;

    const DIMS1: [usize; 1] = [3];
    const DIMS2: [usize; 2] = [3, 1];
    const DIMS3: [usize; 3] = [3, 1, 4];
    const DIMS4: [usize; 4] = [3, 1, 4, 1];
    const DIMS5: [usize; 5] = [3, 1, 4, 1, 5];
    const DIMS6: [usize; 6] = [3, 1, 4, 1, 5, 9];
    const DIMS7: [usize; 7] = [3, 1, 4, 1, 5, 9, 2];
    const DIMS8: [usize; 8] = [3, 1, 4, 1, 5, 9, 2, 6];

    #[test]
    fn test_inner_values() {
        let shape0 = Shape::new([]);
        let shape1 = Shape::new(DIMS1);
        let shape2 = Shape::new(DIMS2);
        let shape3 = Shape::new(DIMS3);
        let shape4 = Shape::new(DIMS4);
        let shape5 = Shape::new(DIMS5);
        let shape6 = Shape::new(DIMS6);
        let shape7 = Shape::new(DIMS7);
        let shape8 = Shape::new(DIMS8);

        assert_eq!(shape0.num_dimensions, 0);
        assert_eq!(shape1.num_dimensions, 1);
        assert_eq!(shape2.num_dimensions, 2);
        assert_eq!(shape3.num_dimensions, 3);
        assert_eq!(shape4.num_dimensions, 4);
        assert_eq!(shape5.num_dimensions, 5);
        assert_eq!(shape6.num_dimensions, 6);
        assert_eq!(shape7.num_dimensions, 7);
        assert_eq!(shape8.num_dimensions, 8);

        assert_eq!(shape0.dimensions, [0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(shape1.dimensions, [3, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(shape2.dimensions, [3, 1, 0, 0, 0, 0, 0, 0]);
        assert_eq!(shape3.dimensions, [3, 1, 4, 0, 0, 0, 0, 0]);
        assert_eq!(shape4.dimensions, [3, 1, 4, 1, 0, 0, 0, 0]);
        assert_eq!(shape5.dimensions, [3, 1, 4, 1, 5, 0, 0, 0]);
        assert_eq!(shape6.dimensions, [3, 1, 4, 1, 5, 9, 0, 0]);
        assert_eq!(shape7.dimensions, [3, 1, 4, 1, 5, 9, 2, 0]);
        assert_eq!(shape8.dimensions, [3, 1, 4, 1, 5, 9, 2, 6]);

        assert_eq!(shape0.num_elements, 1);
        assert_eq!(shape1.num_elements, 3);
        assert_eq!(shape2.num_elements, 3);
        assert_eq!(shape3.num_elements, 12);
        assert_eq!(shape4.num_elements, 12);
        assert_eq!(shape5.num_elements, 60);
        assert_eq!(shape6.num_elements, 540);
        assert_eq!(shape7.num_elements, 1080);
        assert_eq!(shape8.num_elements, 6480);
    }

    #[test]
    #[should_panic]
    fn test_overrun() {
        Shape::new([0; 9]);
    }

    #[test]
    fn test_num_dimensions() {
        assert_eq!(Shape::new([]).num_dimensions(), 0);
        assert_eq!(Shape::new(DIMS1).num_dimensions(), 1);
        assert_eq!(Shape::new(DIMS2).num_dimensions(), 2);
        assert_eq!(Shape::new(DIMS3).num_dimensions(), 3);
        assert_eq!(Shape::new(DIMS4).num_dimensions(), 4);
        assert_eq!(Shape::new(DIMS5).num_dimensions(), 5);
        assert_eq!(Shape::new(DIMS6).num_dimensions(), 6);
        assert_eq!(Shape::new(DIMS7).num_dimensions(), 7);
        assert_eq!(Shape::new(DIMS8).num_dimensions(), 8);
    }

    #[test]
    fn test_check_index() {
        assert!(Shape::new([]).check_index(0).is_err());

        assert!(Shape::new(DIMS1).check_index(0).is_ok());
        assert!(Shape::new(DIMS1).check_index(1).is_err());

        assert!(Shape::new(DIMS2).check_index(0).is_ok());
        assert!(Shape::new(DIMS2).check_index(1).is_ok());
        assert!(Shape::new(DIMS2).check_index(2).is_err());

        assert!(Shape::new(DIMS8).check_index(0).is_ok());
        assert!(Shape::new(DIMS8).check_index(7).is_ok());
        assert!(Shape::new(DIMS8).check_index(8).is_err());
    }

    #[test]
    fn test_check_is_scalar() {
        assert!(Shape::new([]).check_is_scalar().is_ok());
        assert!(Shape::new(DIMS1).check_is_scalar().is_err());
        assert!(Shape::new(DIMS2).check_is_scalar().is_err());
        assert!(Shape::new(DIMS8).check_is_scalar().is_err());
    }

    #[test]
    fn test_dimension_unchecked() {
        unsafe {
            assert_eq!(Shape::new(DIMS1).dimension_unchecked(0), 3);

            assert_eq!(Shape::new(DIMS2).dimension_unchecked(0), 3);
            assert_eq!(Shape::new(DIMS2).dimension_unchecked(1), 1);

            assert_eq!(Shape::new(DIMS8).dimension_unchecked(0), 3);
            assert_eq!(Shape::new(DIMS8).dimension_unchecked(7), 6);
        }
    }

    #[test]
    fn test_dimension() {
        assert!(Shape::new([]).dimension(0).is_err());

        assert_eq!(Shape::new(DIMS1).dimension(0), Ok(3));
        assert!(Shape::new(DIMS1).dimension(1).is_err());

        assert_eq!(Shape::new(DIMS2).dimension(0), Ok(3));
        assert_eq!(Shape::new(DIMS2).dimension(1), Ok(1));
        assert!(Shape::new(DIMS2).dimension(2).is_err());

        assert_eq!(Shape::new(DIMS8).dimension(0), Ok(3));
        assert_eq!(Shape::new(DIMS8).dimension(7), Ok(6));
        assert!(Shape::new(DIMS8).dimension(8).is_err());
    }

    #[test]
    fn test_num_elements() {
        assert_eq!(Shape::new([]).num_elements(), 1);

        assert_eq!(Shape::new([0]).num_elements(), 0);
        assert_eq!(Shape::new([3]).num_elements(), 3);

        assert_eq!(Shape::new([0, 0]).num_elements(), 0);
        assert_eq!(Shape::new([0, 3]).num_elements(), 0);
        assert_eq!(Shape::new([3, 0]).num_elements(), 0);
        assert_eq!(Shape::new([3, 3]).num_elements(), 9);

        assert_eq!(Shape::new([0, 0, 0, 0, 0, 0, 0, 0]).num_elements(), 0);
        assert_eq!(Shape::new([0, 3, 3, 3, 3, 3, 3, 3]).num_elements(), 0);
        assert_eq!(Shape::new([3, 0, 3, 3, 3, 3, 3, 3]).num_elements(), 0);
        assert_eq!(Shape::new([3, 3, 0, 3, 3, 3, 3, 3]).num_elements(), 0);
        assert_eq!(Shape::new([3, 3, 3, 0, 3, 3, 3, 3]).num_elements(), 0);
        assert_eq!(Shape::new([3, 3, 3, 3, 0, 3, 3, 3]).num_elements(), 0);
        assert_eq!(Shape::new([3, 3, 3, 3, 3, 0, 3, 3]).num_elements(), 0);
        assert_eq!(Shape::new([3, 3, 3, 3, 3, 3, 0, 3]).num_elements(), 0);
        assert_eq!(Shape::new([3, 3, 3, 3, 3, 3, 3, 0]).num_elements(), 0);
        assert_eq!(Shape::new([3, 3, 3, 3, 3, 3, 3, 3]).num_elements(), 6561);
    }

    #[test]
    fn test_memory_size() {
        // 0-dimensional
        assert_eq!(Shape::new([]).memory_size::<bool>(), 1);
        assert_eq!(Shape::new([]).memory_size::<i8>(), 1);
        assert_eq!(Shape::new([]).memory_size::<i16>(), 2);
        assert_eq!(Shape::new([]).memory_size::<i32>(), 4);
        assert_eq!(Shape::new([]).memory_size::<i64>(), 8);
        assert_eq!(Shape::new([]).memory_size::<u8>(), 1);
        assert_eq!(Shape::new([]).memory_size::<u16>(), 2);
        assert_eq!(Shape::new([]).memory_size::<u32>(), 4);
        assert_eq!(Shape::new([]).memory_size::<u64>(), 8);
        assert_eq!(Shape::new([]).memory_size::<f32>(), 4);
        assert_eq!(Shape::new([]).memory_size::<f64>(), 8);

        // 1-dimensional
        assert_eq!(Shape::new([0]).memory_size::<bool>(), 0);
        assert_eq!(Shape::new([0]).memory_size::<i8>(), 0);
        assert_eq!(Shape::new([0]).memory_size::<i16>(), 0);
        assert_eq!(Shape::new([0]).memory_size::<i32>(), 0);
        assert_eq!(Shape::new([0]).memory_size::<i64>(), 0);
        assert_eq!(Shape::new([0]).memory_size::<u8>(), 0);
        assert_eq!(Shape::new([0]).memory_size::<u16>(), 0);
        assert_eq!(Shape::new([0]).memory_size::<u32>(), 0);
        assert_eq!(Shape::new([0]).memory_size::<u64>(), 0);
        assert_eq!(Shape::new([0]).memory_size::<f32>(), 0);
        assert_eq!(Shape::new([0]).memory_size::<f64>(), 0);

        assert_eq!(Shape::new([3]).memory_size::<bool>(), 3);
        assert_eq!(Shape::new([3]).memory_size::<i8>(), 3);
        assert_eq!(Shape::new([3]).memory_size::<i16>(), 6);
        assert_eq!(Shape::new([3]).memory_size::<i32>(), 12);
        assert_eq!(Shape::new([3]).memory_size::<i64>(), 24);
        assert_eq!(Shape::new([3]).memory_size::<u8>(), 3);
        assert_eq!(Shape::new([3]).memory_size::<u16>(), 6);
        assert_eq!(Shape::new([3]).memory_size::<u32>(), 12);
        assert_eq!(Shape::new([3]).memory_size::<u64>(), 24);
        assert_eq!(Shape::new([3]).memory_size::<f32>(), 12);
        assert_eq!(Shape::new([3]).memory_size::<f64>(), 24);

        // 2-dimensional, only for f32
        assert_eq!(Shape::new([0, 0]).memory_size::<f32>(), 0);
        assert_eq!(Shape::new([0, 3]).memory_size::<f32>(), 0);
        assert_eq!(Shape::new([3, 0]).memory_size::<f32>(), 0);
        assert_eq!(Shape::new([3, 3]).memory_size::<f32>(), 36);

        // 8-dimensional, only for f32
        assert_eq!(Shape::new([0, 0, 0, 0, 0, 0, 0, 0]).memory_size::<f32>(), 0);
        assert_eq!(Shape::new([0, 3, 3, 3, 3, 3, 3, 3]).memory_size::<f32>(), 0);
        assert_eq!(Shape::new([3, 0, 3, 3, 3, 3, 3, 3]).memory_size::<f32>(), 0);
        assert_eq!(Shape::new([3, 3, 0, 3, 3, 3, 3, 3]).memory_size::<f32>(), 0);
        assert_eq!(Shape::new([3, 3, 3, 0, 3, 3, 3, 3]).memory_size::<f32>(), 0);
        assert_eq!(Shape::new([3, 3, 3, 3, 0, 3, 3, 3]).memory_size::<f32>(), 0);
        assert_eq!(Shape::new([3, 3, 3, 3, 3, 0, 3, 3]).memory_size::<f32>(), 0);
        assert_eq!(Shape::new([3, 3, 3, 3, 3, 3, 0, 3]).memory_size::<f32>(), 0);
        assert_eq!(Shape::new([3, 3, 3, 3, 3, 3, 3, 0]).memory_size::<f32>(), 0);
        assert_eq!(
            Shape::new([3, 3, 3, 3, 3, 3, 3, 3]).memory_size::<f32>(),
            4 * 6561
        );
    }

    #[rustfmt::skip]
    #[test]
    fn test_elementwise() {
        assert_eq!(Shape::new([]).elementwise(&Shape::new([])), Ok(Shape::new([])));
        assert_eq!(Shape::new(DIMS1).elementwise(&Shape::new(DIMS1)), Ok(Shape::new(DIMS1)));
        assert_eq!(Shape::new(DIMS2).elementwise(&Shape::new(DIMS2)), Ok(Shape::new(DIMS2)));
        assert_eq!(Shape::new(DIMS3).elementwise(&Shape::new(DIMS3)), Ok(Shape::new(DIMS3)));
        assert_eq!(Shape::new(DIMS4).elementwise(&Shape::new(DIMS4)), Ok(Shape::new(DIMS4)));
        assert_eq!(Shape::new(DIMS5).elementwise(&Shape::new(DIMS5)), Ok(Shape::new(DIMS5)));
        assert_eq!(Shape::new(DIMS6).elementwise(&Shape::new(DIMS6)), Ok(Shape::new(DIMS6)));
        assert_eq!(Shape::new(DIMS7).elementwise(&Shape::new(DIMS7)), Ok(Shape::new(DIMS7)));
        assert_eq!(Shape::new(DIMS8).elementwise(&Shape::new(DIMS8)), Ok(Shape::new(DIMS8)));

        // Different number of dimensions.
        assert!(Shape::new([]).elementwise(&Shape::new(DIMS1)).is_err());
        assert!(Shape::new([]).elementwise(&Shape::new(DIMS2)).is_err());
        assert!(Shape::new(DIMS1).elementwise(&Shape::new([])).is_err());
        assert!(Shape::new(DIMS1).elementwise(&Shape::new(DIMS2)).is_err());
        assert!(Shape::new(DIMS2).elementwise(&Shape::new([])).is_err());
        assert!(Shape::new(DIMS2).elementwise(&Shape::new(DIMS1)).is_err());

        // Same number of dimensions, but each dimension has different size.
        // Note that the elementwise operations does not perform reshaping and broadcasting.
        assert!(Shape::new([0]).elementwise(&Shape::new([1])).is_err());
        assert!(Shape::new([0]).elementwise(&Shape::new([3])).is_err());
        assert!(Shape::new([1]).elementwise(&Shape::new([0])).is_err());
        assert!(Shape::new([1]).elementwise(&Shape::new([3])).is_err());
        assert!(Shape::new([3]).elementwise(&Shape::new([0])).is_err());
        assert!(Shape::new([3]).elementwise(&Shape::new([1])).is_err());
        assert!(Shape::new([3]).elementwise(&Shape::new([42])).is_err());

        assert!(Shape::new([2, 3]).elementwise(&Shape::new([0, 3])).is_err());
        assert!(Shape::new([2, 3]).elementwise(&Shape::new([1, 3])).is_err());
        assert!(Shape::new([2, 3]).elementwise(&Shape::new([42, 3])).is_err());
        assert!(Shape::new([2, 3]).elementwise(&Shape::new([2, 0])).is_err());
        assert!(Shape::new([2, 3]).elementwise(&Shape::new([2, 1])).is_err());
        assert!(Shape::new([2, 3]).elementwise(&Shape::new([2, 42])).is_err());

        assert!(Shape::new([2, 3]).elementwise(&Shape::new([1, 6])).is_err());
        assert!(Shape::new([2, 3]).elementwise(&Shape::new([3, 2])).is_err());
        assert!(Shape::new([2, 3]).elementwise(&Shape::new([6, 1])).is_err());
    }

    #[test]
    fn test_fmt() {
        assert_eq!(format!("{}", Shape::new([])), "()");
        assert_eq!(format!("{}", Shape::new(DIMS1)), "(3)");
        assert_eq!(format!("{}", Shape::new(DIMS2)), "(3, 1)");
        assert_eq!(format!("{}", Shape::new(DIMS3)), "(3, 1, 4)");
        assert_eq!(format!("{}", Shape::new(DIMS4)), "(3, 1, 4, 1)");
        assert_eq!(format!("{}", Shape::new(DIMS5)), "(3, 1, 4, 1, 5)");
        assert_eq!(format!("{}", Shape::new(DIMS6)), "(3, 1, 4, 1, 5, 9)");
        assert_eq!(format!("{}", Shape::new(DIMS7)), "(3, 1, 4, 1, 5, 9, 2)");
        assert_eq!(format!("{}", Shape::new(DIMS8)), "(3, 1, 4, 1, 5, 9, 2, 6)");
    }
}

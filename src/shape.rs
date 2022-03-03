use crate::error::Error;
use crate::result::Result;
use std::fmt;
use std::mem::{size_of, MaybeUninit};

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
    /// This function takes a fixed-length array so that the compiler is expected to generate
    /// highly-optimized machine code for the particular value of `N`.
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
        assert!(
            N <= MAX_NUM_DIMENSIONS,
            "Number of dimensions must be equal to or less than {}, but got {}.",
            MAX_NUM_DIMENSIONS,
            N
        );
        let mut actual_dimensions = [0usize; MAX_NUM_DIMENSIONS];
        let mut num_elements = 1usize;
        for (ad, d) in actual_dimensions.iter_mut().zip(dimensions.iter()) {
            *ad = *d;
            num_elements *= d;
        }
        Self {
            num_dimensions: N,
            dimensions: actual_dimensions,
            num_elements,
        }
    }

    /// Creates a new n-dimensional shape from a slice.
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
    /// The length of the `dimensions` is larger than `MAX_NUM_DIMENSIONS`.
    pub fn from_slice(dimensions: &[usize]) -> Self {
        let num_dimensions = dimensions.len();
        assert!(
            num_dimensions <= MAX_NUM_DIMENSIONS,
            "Number of dimensions must be equal to or less than {}, but got {}.",
            MAX_NUM_DIMENSIONS,
            num_dimensions
        );
        let mut actual_dimensions = [0usize; MAX_NUM_DIMENSIONS];
        let mut num_elements = 1usize;
        for (ad, d) in actual_dimensions.iter_mut().zip(dimensions.iter()) {
            *ad = *d;
            num_elements *= d;
        }
        Self {
            num_dimensions,
            dimensions: actual_dimensions,
            num_elements,
        }
    }

    /// Obtains dimensions as an array.
    ///
    /// # Returns
    ///
    /// * `Ok([usize; N])` - Array representation of dimensions.
    /// * `Err(Error)` - The shape is not N-dimensional.
    pub fn as_array<const N: usize>(&self) -> Result<[usize; N]> {
        if N == self.num_dimensions {
            let mut v = MaybeUninit::<[usize; N]>::uninit();
            unsafe { &mut *v.as_mut_ptr() }.copy_from_slice(&self.dimensions[..N]);
            Ok(unsafe { v.assume_init() })
        } else {
            Err(Error::InvalidLength(format!(
                "Requested dimensions of length {}, but the shape is {}-dimensional",
                N, self.num_dimensions
            )))
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
mod tests;

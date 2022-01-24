use crate::buffer::Buffer;
use crate::hardware::get_default_hardware;
use crate::make_shape;
use crate::result::Result;
use crate::shape::Shape;
use std::mem::size_of;

/// A multidimensional array with specific computing backend.
///
/// This structure abstracts most of hardware implementations and provides user-level operations
/// for array data.
pub struct Array {
    /// Shape of this array.
    shape: Shape,

    /// Buffer of the data.
    buffer: Buffer<'static>,
}

impl Array {
    /// Creates a new Array on the default hardware.
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape of the array.
    ///
    /// # Returns
    ///
    /// A new `Array` object.
    pub(crate) fn with_default_hardware(shape: Shape) -> Self {
        Self {
            shape,
            buffer: Buffer::new(
                get_default_hardware(),
                shape.get_num_elements() * size_of::<f32>(),
            ),
        }
    }

    /// Returns the shape of the array.
    ///
    /// # Returns
    ///
    /// Reference to the `Shape` object.
    ///
    /// Shape of the array.
    pub(crate) fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Sets a scalar value to this array.
    ///
    /// # Arguments
    ///
    /// * `value` - New value to set.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - New value is set correctly.
    /// * `Err(Error)` - Array is not a scalar.
    pub(crate) fn set_scalar(&mut self, value: f32) -> Result<()> {
        self.shape.check_is_scalar()?;
        unsafe {
            self.buffer
                .hardware()
                .lock()
                .unwrap()
                .copy_host_to_hardware(
                    (&value as *const f32) as *const u8,
                    self.buffer.as_handle_mut(),
                    size_of::<f32>(),
                )
        }
        Ok(())
    }

    /// Obtains scalar value of this array.
    ///
    /// # Returns
    ///
    /// Scalar value of the array.
    ///
    /// # Returns
    ///
    /// * `Ok(f32)` - Scalar value obtained from the array.
    /// * `Err(Error)` - Array is not a scalar.
    pub(crate) fn into_scalar(&self) -> Result<f32> {
        self.shape.check_is_scalar()?;
        let mut value: f32;
        unsafe {
            self.buffer
                .hardware()
                .lock()
                .unwrap()
                .copy_hardware_to_host(
                    self.buffer.as_handle(),
                    (&mut value as *mut f32) as *mut u8,
                    size_of::<f32>(),
                )
        }
        Ok(value)
    }
}

/// For early-stage debugging, will be removed.
pub(crate) fn make_cpu_scalar(value: f32) -> Array {
    let array = Array::with_default_hardware(make_shape![]);
    array.set_scalar(value);
    array
}

#[cfg(test)]
mod tests {
    use crate::array::make_cpu_scalar;
    use crate::make_shape;

    #[test]
    fn test_cpu_scalar() {
        let value = make_cpu_scalar(123.);
        assert_eq!(value.shape(), &make_shape![]);
        assert_eq!(value.into_scalar(), Ok(123.));
    }
}

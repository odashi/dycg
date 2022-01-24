use crate::buffer::Buffer;
use crate::hardware::{get_default_hardware, Hardware};
use crate::make_shape;
use crate::result::Result;
use crate::shape::Shape;
use std::mem::size_of;
use std::sync::Mutex;

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
    /// Creates a new `Array` on a specific hardware.
    ///
    /// # Arguments
    ///
    /// * `hardware` - Hardware that handles the memory.
    /// * `shape` - `Shape` of the new array.
    ///
    /// # Returns
    ///
    /// A new `Array` object.
    ///
    /// # Safety
    ///
    /// This function does not initialize the inner memory.
    /// Users are responsible to initialize the memory immediately by themselves.
    pub(crate) unsafe fn raw(hardware: &'static Mutex<Box<dyn Hardware>>, shape: Shape) -> Self {
        Self {
            shape,
            buffer: Buffer::raw(hardware, shape.get_memory_size::<f32>()),
        }
    }

    /// Creates a new `Array` on the same hardware with `other`.
    ///
    /// # Arguments
    ///
    /// * `other` - An `Array` object on the desired hardware.
    /// * `shape` - `Shape` of the new array.
    ///
    /// # Returns
    ///
    /// A new `Array` object.
    ///
    /// # Safety
    ///
    /// This function does not initialize the inner memory.
    /// Users are responsible to initialize the memory immediately by themselves.
    pub(crate) unsafe fn raw_colocated(other: &Array, shape: Shape) -> Self {
        Self {
            shape,
            buffer: Buffer::raw_colocated(&other.buffer, shape.get_memory_size::<f32>()),
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
    unsafe {
        let mut array = Array::raw(get_default_hardware(), make_shape![]);
        array.set_scalar(value);
        array
    }
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

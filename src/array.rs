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
        let mut value = 0.;
        unsafe {
            self.buffer
                .hardware()
                .lock()
                .unwrap()
                .copy_hardware_to_host(
                    self.buffer.as_handle(),
                    (&mut value as *mut f32) as *mut u8,
                    size_of::<f32>(),
                );
        }
        Ok(value)
    }

    /// Performs elementwise add operation and returns a new `Array` of resulting values.
    ///
    /// This function does not perform broadcasting.
    ///
    /// # Arguments
    ///
    /// * `other` - `Array` of right-hand side argument.
    ///
    /// # Returns
    ///
    /// * `Ok(Array)` - A new `Array` holding the results.
    /// * `Err(Error)` - The operation can not be evaluated for given arguments.
    pub(crate) fn elementwise_add_f32(&self, other: &Array) -> Result<Array> {
        self.buffer.check_colocated(&other.buffer)?;
        let output_shape = self.shape.elementwise(&other.shape)?;
        unsafe {
            let mut output = Self::raw_colocated(self, output_shape);
            output
                .buffer
                .hardware()
                .lock()
                .unwrap()
                .elementwise_add_f32(
                    self.buffer.as_handle(),
                    other.buffer.as_handle(),
                    output.buffer.as_handle_mut(),
                    output_shape.get_num_elements(),
                );
            Ok(output)
        }
    }

    /// Performs elementwise subtract operation and returns a new `Array` of resulting values.
    ///
    /// This function does not perform broadcasting.
    ///
    /// # Arguments
    ///
    /// * `other` - `Array` of right-hand side argument.
    ///
    /// # Returns
    ///
    /// * `Ok(Array)` - A new `Array` holding the results.
    /// * `Err(Error)` - The operation can not be evaluated for given arguments.
    pub(crate) fn elementwise_sub_f32(&self, other: &Array) -> Result<Array> {
        self.buffer.check_colocated(&other.buffer)?;
        let output_shape = self.shape.elementwise(&other.shape)?;
        unsafe {
            let mut output = Self::raw_colocated(self, output_shape);
            output
                .buffer
                .hardware()
                .lock()
                .unwrap()
                .elementwise_sub_f32(
                    self.buffer.as_handle(),
                    other.buffer.as_handle(),
                    output.buffer.as_handle_mut(),
                    output_shape.get_num_elements(),
                );
            Ok(output)
        }
    }

    /// Performs elementwise multiply operation and returns a new `Array` of resulting values.
    ///
    /// This function does not perform broadcasting.
    ///
    /// # Arguments
    ///
    /// * `other` - `Array` of right-hand side argument.
    ///
    /// # Returns
    ///
    /// * `Ok(Array)` - A new `Array` holding the results.
    /// * `Err(Error)` - The operation can not be evaluated for given arguments.
    pub(crate) fn elementwise_mul_f32(&self, other: &Array) -> Result<Array> {
        self.buffer.check_colocated(&other.buffer)?;
        let output_shape = self.shape.elementwise(&other.shape)?;
        unsafe {
            let mut output = Self::raw_colocated(self, output_shape);
            output
                .buffer
                .hardware()
                .lock()
                .unwrap()
                .elementwise_mul_f32(
                    self.buffer.as_handle(),
                    other.buffer.as_handle(),
                    output.buffer.as_handle_mut(),
                    output_shape.get_num_elements(),
                );
            Ok(output)
        }
    }

    /// Performs elementwise divide operation and returns a new `Array` of resulting values.
    ///
    /// This function does not perform broadcasting.
    ///
    /// # Arguments
    ///
    /// * `other` - `Array` of right-hand side argument.
    ///
    /// # Returns
    ///
    /// * `Ok(Array)` - A new `Array` holding the results.
    /// * `Err(Error)` - The operation can not be evaluated for given arguments.
    pub(crate) fn elementwise_div_f32(&self, other: &Array) -> Result<Array> {
        self.buffer.check_colocated(&other.buffer)?;
        let output_shape = self.shape.elementwise(&other.shape)?;
        unsafe {
            let mut output = Self::raw_colocated(self, output_shape);
            output
                .buffer
                .hardware()
                .lock()
                .unwrap()
                .elementwise_div_f32(
                    self.buffer.as_handle(),
                    other.buffer.as_handle(),
                    output.buffer.as_handle_mut(),
                    output_shape.get_num_elements(),
                );
            Ok(output)
        }
    }
}

/// For early-stage debugging, will be removed.
pub(crate) fn make_cpu_scalar(value: f32) -> Array {
    unsafe {
        let mut array = Array::raw(get_default_hardware(), make_shape![]);
        array.set_scalar(value).unwrap();
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

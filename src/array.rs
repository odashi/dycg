use crate::buffer::Buffer;
use crate::error::Error;
use crate::hardware::Hardware;
use crate::make_shape;
use crate::result::Result;
use crate::shape::Shape;
use std::cell::RefCell;
use std::mem;

/// A multidimensional array with specific computing backend.
///
/// This structure abstracts most of hardware implementations and provides user-level operations
/// for array data.
pub struct Array<'hw> {
    /// Shape of this array.
    shape: Shape,

    /// Buffer of the data.
    buffer: Buffer<'hw>,
}

impl<'hw> Array<'hw> {
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
    unsafe fn raw(hardware: &'hw RefCell<dyn Hardware>, shape: Shape) -> Self {
        let size = shape.memory_size::<f32>();
        Self {
            shape,
            buffer: Buffer::raw(hardware, size),
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
    unsafe fn raw_colocated(other: &Self, shape: Shape) -> Self {
        let size = shape.memory_size::<f32>();
        Self {
            shape,
            buffer: Buffer::raw_colocated(&other.buffer, size),
        }
    }

    /// Returns the shape of the array.
    ///
    /// # Returns
    ///
    /// Reference to the `Shape` object.
    ///
    /// Shape of the array.
    pub fn shape(&self) -> &Shape {
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
    pub fn set_scalar_f32(&mut self, value: f32) -> Result<()> {
        self.shape.check_is_scalar()?;
        unsafe {
            self.buffer.hardware().borrow_mut().copy_host_to_hardware(
                (&value as *const f32) as *const u8,
                self.buffer.as_mut_handle(),
                mem::size_of::<f32>(),
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
    pub fn get_scalar_f32(&self) -> Result<f32> {
        self.shape.check_is_scalar()?;
        let mut value = 0.;
        unsafe {
            self.buffer.hardware().borrow_mut().copy_hardware_to_host(
                self.buffer.as_handle(),
                (&mut value as *mut f32) as *mut u8,
                mem::size_of::<f32>(),
            );
        }
        Ok(value)
    }

    /// Sets all values in the underlying buffer.
    ///
    /// # Arguments
    ///
    /// * `values` - Sequence of values to be set. The length must be the same as the size of
    ///   the underlying buffer.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Values are set correctly.
    /// * `Err(Error)` - Length of the specified values is different with the size of the
    ///   underlying buffer.
    pub fn set_values_f32(&mut self, values: &[f32]) -> Result<()> {
        if values.len() != self.shape.num_elements() {
            return Err(Error::InvalidLength(format!(
                "Values has a different length. Required {}, but got {}.",
                self.shape.num_elements(),
                values.len(),
            )));
        }

        unsafe {
            self.buffer.hardware().borrow_mut().copy_host_to_hardware(
                values.as_ptr() as *const u8,
                self.buffer.as_mut_handle(),
                self.shape.num_elements() * mem::size_of::<f32>(),
            );
        }

        Ok(())
    }

    /// Obtains all values in the underlying buffer.
    ///
    /// # Returns
    ///
    /// `Vec` of all values. The order of values is row-major order (C order).
    pub fn get_values_f32(&self) -> Vec<f32> {
        let num_elements = self.shape.num_elements();
        let mut values = Vec::<f32>::with_capacity(num_elements);
        unsafe {
            self.buffer.hardware().borrow_mut().copy_hardware_to_host(
                self.buffer.as_handle(),
                values.as_mut_ptr() as *mut u8,
                num_elements * mem::size_of::<f32>(),
            );
            values.set_len(num_elements);
        }
        values
    }

    /// Creates a new `Array` with 0-dimensional shape on the specific hardware.
    ///
    /// # Arguments
    ///
    /// * `hardware`: `Hardware` object to host the value.
    /// * `value`: Value of the resulting array.
    ///
    /// # Returns
    ///
    /// A new `Array` object representing a scalar value.
    pub fn scalar_f32(hardware: &'hw RefCell<dyn Hardware>, value: f32) -> Self {
        let mut array = unsafe { Self::raw(hardware, make_shape![]) };
        array.set_scalar_f32(value).unwrap();
        array
    }

    /// Creates a new `Array` with arbitrary shape on the specific hardware.
    ///
    /// # Arguments
    ///
    /// * `hardware`: `Hardware` object to host the value.
    /// * `shape`: `Shape` of the new `Array`.
    /// * `values`: Values to be copied to the underlying buffer. The number of values must be the same as
    ///   the size of the underlying buffer.
    ///
    /// # Returns
    ///
    /// * `Ok(Array)` - A new `Array` object.
    /// * Err(Error)` - Some error occurred during the process.
    pub fn constant_f32(
        hardware: &'hw RefCell<dyn Hardware>,
        shape: Shape,
        values: &[f32],
    ) -> Result<Self> {
        let mut array = unsafe { Self::raw(hardware, shape) };
        array.set_values_f32(values)?;
        Ok(array)
    }

    /// Creates a new `Array` with a specified `Shape` filled by a single value.
    ///
    /// # Arguments
    ///
    /// * `hardware`: `Hardware` object to host the value.
    /// * `shape` - `Shape` of the resulting `Array`.
    /// * `value` - Value of the elements.
    ///
    /// # Returns
    ///
    /// * `Ok(Array)` - A new `Array` object.
    /// * `Err(Error)` - Some error occurred during the process.
    pub fn fill_f32(hardware: &'hw RefCell<dyn Hardware>, shape: Shape, value: f32) -> Self {
        unsafe {
            let mut array = Self::raw(hardware, shape);
            hardware.borrow_mut().fill_f32(
                array.buffer.as_mut_handle(),
                value,
                array.shape.num_elements(),
            );
            array
        }
    }

    /// Creates a new `Array` with a specified `Shape` filled by a single value, colocated with an existing `Array`.
    ///
    /// # Arguments
    ///
    /// * `other` - An `Array` object on the desired hardware.
    /// * `shape` - `Shape` of the resulting `Array`.
    /// * `value` - Value of the elements.
    ///
    /// # Returns
    ///
    /// * `Ok(Array)` - A new `Array` object.
    /// * `Err(Error)` - Some error occurred during the process.
    pub fn fill_colocated_f32(other: &Self, shape: Shape, value: f32) -> Self {
        Self::fill_f32(other.buffer.hardware(), shape, value)
    }

    /// Performs elementwise negation operation and returns a new `Array` of resulting
    /// values.
    ///
    /// This function does not perform broadcasting.
    ///
    /// # Returns
    ///
    /// * `Ok(Array)` - A new `Array` holding the results.
    /// * `Err(Error)` - The operation can not be evaluated for given arguments.
    pub fn elementwise_neg_f32(&self) -> Self {
        unsafe {
            let mut output = Self::raw_colocated(self, self.shape.clone());
            output.buffer.hardware().borrow_mut().elementwise_neg_f32(
                self.buffer.as_handle(),
                output.buffer.as_mut_handle(),
                self.shape.num_elements(),
            );
            output
        }
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
    pub fn elementwise_add_f32(&self, other: &Self) -> Result<Self> {
        self.buffer.check_colocated(&other.buffer)?;
        let output_shape = self.shape.elementwise(&other.shape)?;
        let num_elements = output_shape.num_elements();
        unsafe {
            let mut output = Self::raw_colocated(self, output_shape);
            output.buffer.hardware().borrow_mut().elementwise_add_f32(
                self.buffer.as_handle(),
                other.buffer.as_handle(),
                output.buffer.as_mut_handle(),
                num_elements,
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
    pub fn elementwise_sub_f32(&self, other: &Self) -> Result<Self> {
        self.buffer.check_colocated(&other.buffer)?;
        let output_shape = self.shape.elementwise(&other.shape)?;
        let num_elements = output_shape.num_elements();
        unsafe {
            let mut output = Self::raw_colocated(self, output_shape);
            output.buffer.hardware().borrow_mut().elementwise_sub_f32(
                self.buffer.as_handle(),
                other.buffer.as_handle(),
                output.buffer.as_mut_handle(),
                num_elements,
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
    pub fn elementwise_mul_f32(&self, other: &Self) -> Result<Self> {
        self.buffer.check_colocated(&other.buffer)?;
        let output_shape = self.shape.elementwise(&other.shape)?;
        let num_elements = output_shape.num_elements();
        unsafe {
            let mut output = Self::raw_colocated(self, output_shape);
            output.buffer.hardware().borrow_mut().elementwise_mul_f32(
                self.buffer.as_handle(),
                other.buffer.as_handle(),
                output.buffer.as_mut_handle(),
                num_elements,
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
    pub fn elementwise_div_f32(&self, other: &Self) -> Result<Self> {
        self.buffer.check_colocated(&other.buffer)?;
        let output_shape = self.shape.elementwise(&other.shape)?;
        let num_elements = output_shape.num_elements();
        unsafe {
            let mut output = Self::raw_colocated(self, output_shape);
            output.buffer.hardware().borrow_mut().elementwise_div_f32(
                self.buffer.as_handle(),
                other.buffer.as_handle(),
                output.buffer.as_mut_handle(),
                num_elements,
            );
            Ok(output)
        }
    }
}

impl<'hw> Clone for Array<'hw> {
    fn clone(&self) -> Self {
        unsafe {
            let mut cloned = Self::raw_colocated(self, self.shape().clone());
            cloned
                .buffer
                .hardware()
                .borrow_mut()
                .copy_hardware_to_hardware(
                    self.buffer.as_handle(),
                    cloned.buffer.as_mut_handle(),
                    self.shape().memory_size::<f32>(),
                );
            cloned
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::array::Array;
    use crate::hardware::cpu::CpuHardware;
    use crate::make_shape;
    use std::cell::RefCell;
    use std::mem::size_of;
    use std::ptr;

    #[test]
    fn test_raw_scalar() {
        let hw = RefCell::new(CpuHardware::new());
        let array = unsafe { Array::raw(&hw, make_shape![]) };
        assert!(ptr::eq(array.buffer.hardware(), &hw));
        assert_eq!(array.buffer.size(), size_of::<f32>());
        assert_eq!(array.shape, make_shape![]);
    }

    #[test]
    fn test_raw_0() {
        let hw = RefCell::new(CpuHardware::new());
        let array = unsafe { Array::raw(&hw, make_shape![0]) };
        assert!(ptr::eq(array.buffer.hardware(), &hw));
        assert_eq!(array.buffer.size(), 0);
        assert_eq!(array.shape, make_shape![0]);
    }

    #[test]
    fn test_raw_n() {
        let hw = RefCell::new(CpuHardware::new());
        let array = unsafe { Array::raw(&hw, make_shape![42]) };
        assert!(ptr::eq(array.buffer.hardware(), &hw));
        assert_eq!(array.buffer.size(), 42 * size_of::<f32>());
        assert_eq!(array.shape, make_shape![42]);
    }

    #[test]
    fn test_raw_colocated_scalar() {
        let hw = RefCell::new(CpuHardware::new());
        let other = unsafe { Array::raw(&hw, make_shape![]) };
        let colocated = unsafe { Array::raw_colocated(&other, make_shape![]) };
        assert!(ptr::eq(colocated.buffer.hardware(), &hw));
        assert_eq!(colocated.buffer.size(), size_of::<f32>());
        assert_eq!(colocated.shape, make_shape![]);
    }

    #[test]
    fn test_raw_colocated_0() {
        let hw = RefCell::new(CpuHardware::new());
        let other = unsafe { Array::raw(&hw, make_shape![]) };
        let colocated = unsafe { Array::raw_colocated(&other, make_shape![0]) };
        assert!(ptr::eq(colocated.buffer.hardware(), &hw));
        assert_eq!(colocated.buffer.size(), 0);
        assert_eq!(colocated.shape, make_shape![0]);
    }

    #[test]
    fn test_raw_colocated_n() {
        let hw = RefCell::new(CpuHardware::new());
        let other = unsafe { Array::raw(&hw, make_shape![]) };
        let colocated = unsafe { Array::raw_colocated(&other, make_shape![42]) };
        assert!(ptr::eq(colocated.buffer.hardware(), &hw));
        assert_eq!(colocated.buffer.size(), 42 * size_of::<f32>());
        assert_eq!(colocated.shape, make_shape![42]);
    }

    #[test]
    fn test_shape() {
        let hw = RefCell::new(CpuHardware::new());
        let array = unsafe { Array::raw(&hw, make_shape![]) };
        assert!(ptr::eq(array.shape(), &array.shape));
    }

    #[test]
    fn test_set_scalar_f32_scalar() {
        let hw = RefCell::new(CpuHardware::new());
        let mut array = unsafe { Array::raw(&hw, make_shape![]) };
        array.set_scalar_f32(123.).unwrap();
        assert_eq!(array.get_scalar_f32(), Ok(123.));
        assert_eq!(array.get_values_f32(), vec![123.]);
    }

    #[test]
    fn test_set_scalar_f32_1() {
        let hw = RefCell::new(CpuHardware::new());
        let mut array = unsafe { Array::raw(&hw, make_shape![1]) };
        assert!(array.set_scalar_f32(123.).is_err());
    }

    #[test]
    fn test_set_scalar_f32_n() {
        let hw = RefCell::new(CpuHardware::new());
        let mut array = unsafe { Array::raw(&hw, make_shape![42]) };
        assert!(array.set_scalar_f32(123.).is_err());
    }

    #[test]
    fn test_get_scalar_f32_1() {
        let hw = RefCell::new(CpuHardware::new());
        let array = unsafe { Array::raw(&hw, make_shape![1]) };
        assert!(array.get_scalar_f32().is_err());
    }

    #[test]
    fn test_get_scalar_f32_n() {
        let hw = RefCell::new(CpuHardware::new());
        let array = unsafe { Array::raw(&hw, make_shape![42]) };
        assert!(array.get_scalar_f32().is_err());
    }

    #[test]
    fn test_set_values_f32_scalar() {
        let hw = RefCell::new(CpuHardware::new());
        let mut array = unsafe { Array::raw(&hw, make_shape![]) };
        array.set_values_f32(&[123.]).unwrap();
        assert_eq!(array.get_scalar_f32(), Ok(123.));
        assert_eq!(array.get_values_f32(), vec![123.]);

        assert!(array.set_values_f32(&[]).is_err());
        assert!(array.set_values_f32(&[123., 456.]).is_err());
        assert!(array.set_values_f32(&[123., 456., 789.]).is_err());
    }

    #[test]
    fn test_set_values_f32_0() {
        let hw = RefCell::new(CpuHardware::new());
        let mut array = unsafe { Array::raw(&hw, make_shape![0]) };
        array.set_values_f32(&[]).unwrap();
        assert_eq!(array.get_values_f32(), vec![]);

        assert!(array.set_values_f32(&[111.]).is_err());
        assert!(array.set_values_f32(&[111., 222.]).is_err());
    }

    #[test]
    fn test_set_values_f32_n() {
        let hw = RefCell::new(CpuHardware::new());
        let mut array = unsafe { Array::raw(&hw, make_shape![3]) };
        array.set_values_f32(&[123., 456., 789.]).unwrap();
        assert_eq!(array.get_values_f32(), vec![123., 456., 789.]);

        assert!(array.set_values_f32(&[]).is_err());
        assert!(array.set_values_f32(&[111.]).is_err());
        assert!(array.set_values_f32(&[111., 222.]).is_err());
        assert!(array.set_values_f32(&[111., 222., 333., 444.]).is_err());
    }

    #[test]
    fn test_scalar_f32() {
        let hw = RefCell::new(CpuHardware::new());
        let array = Array::scalar_f32(&hw, 123.);
        assert_eq!(array.shape, make_shape![]);
        assert_eq!(array.get_scalar_f32(), Ok(123.));
        assert_eq!(array.get_values_f32(), vec![123.]);
    }

    #[test]
    fn test_constant_f32_scalar() {
        let hw = RefCell::new(CpuHardware::new());
        let array = Array::constant_f32(&hw, make_shape![], &[123.]).unwrap();
        assert_eq!(array.shape, make_shape![]);
        assert_eq!(array.get_scalar_f32(), Ok(123.));
        assert_eq!(array.get_values_f32(), vec![123.]);
    }

    #[test]
    fn test_constant_f32_scalar_invalid() {
        let hw = RefCell::new(CpuHardware::new());
        assert!(Array::constant_f32(&hw, make_shape![], &[]).is_err());
        assert!(Array::constant_f32(&hw, make_shape![], &[111., 222.]).is_err());
    }

    #[test]
    fn test_constant_f32_0() {
        let hw = RefCell::new(CpuHardware::new());
        let array = Array::constant_f32(&hw, make_shape![0], &[]).unwrap();
        assert_eq!(array.shape, make_shape![0]);
        assert!(array.get_scalar_f32().is_err());
        assert_eq!(array.get_values_f32(), vec![]);
    }

    #[test]
    fn test_constant_f32_0_invalid() {
        let hw = RefCell::new(CpuHardware::new());
        assert!(Array::constant_f32(&hw, make_shape![0], &[111.]).is_err());
        assert!(Array::constant_f32(&hw, make_shape![0], &[111., 222.]).is_err());
    }

    #[test]
    fn test_constant_f32_n() {
        let hw = RefCell::new(CpuHardware::new());
        let array = Array::constant_f32(&hw, make_shape![3], &[123., 456., 789.]).unwrap();
        assert_eq!(array.shape, make_shape![3]);
        assert!(array.get_scalar_f32().is_err());
        assert_eq!(array.get_values_f32(), vec![123., 456., 789.]);
    }

    #[test]
    fn test_constant_f32_n_invalid() {
        let hw = RefCell::new(CpuHardware::new());
        assert!(Array::constant_f32(&hw, make_shape![3], &[]).is_err());
        assert!(Array::constant_f32(&hw, make_shape![3], &[111.]).is_err());
        assert!(Array::constant_f32(&hw, make_shape![3], &[111., 222.]).is_err());
        assert!(Array::constant_f32(&hw, make_shape![3], &[111., 222., 333., 444.]).is_err());
    }

    #[test]
    fn test_fill_f32_scalar() {
        let hw = RefCell::new(CpuHardware::new());
        let array = Array::fill_f32(&hw, make_shape![], 123.);
        assert_eq!(array.shape, make_shape![]);
        assert_eq!(array.get_scalar_f32(), Ok(123.));
        assert_eq!(array.get_values_f32(), vec![123.]);
    }

    #[test]
    fn test_fill_f32_0() {
        let hw = RefCell::new(CpuHardware::new());
        let array = Array::fill_f32(&hw, make_shape![0], 123.);
        assert_eq!(array.shape, make_shape![0]);
        assert!(array.get_scalar_f32().is_err());
        assert_eq!(array.get_values_f32(), vec![]);
    }

    #[test]
    fn test_fill_f32_n() {
        let hw = RefCell::new(CpuHardware::new());
        let array = Array::fill_f32(&hw, make_shape![3], 123.);
        assert_eq!(array.shape, make_shape![3]);
        assert!(array.get_scalar_f32().is_err());
        assert_eq!(array.get_values_f32(), vec![123., 123., 123.]);
    }

    #[test]
    fn test_fill_colocated_f32_scalar() {
        let hw = RefCell::new(CpuHardware::new());
        let other = unsafe { Array::raw(&hw, make_shape![]) };
        let array = Array::fill_colocated_f32(&other, make_shape![], 123.);
        assert_eq!(array.shape, make_shape![]);
        assert!(ptr::eq(array.buffer.hardware(), &hw));
        assert_eq!(array.get_scalar_f32(), Ok(123.));
        assert_eq!(array.get_values_f32(), vec![123.]);
    }

    #[test]
    fn test_fill_colocated_f32_0() {
        let hw = RefCell::new(CpuHardware::new());
        let other = unsafe { Array::raw(&hw, make_shape![]) };
        let array = Array::fill_colocated_f32(&other, make_shape![0], 123.);
        assert_eq!(array.shape, make_shape![0]);
        assert!(ptr::eq(array.buffer.hardware(), &hw));
        assert!(array.get_scalar_f32().is_err());
        assert_eq!(array.get_values_f32(), vec![]);
    }

    #[test]
    fn test_fill_colocated_f32_n() {
        let hw = RefCell::new(CpuHardware::new());
        let other = unsafe { Array::raw(&hw, make_shape![]) };
        let array = Array::fill_colocated_f32(&other, make_shape![3], 123.);
        assert_eq!(array.shape, make_shape![3]);
        assert!(ptr::eq(array.buffer.hardware(), &hw));
        assert!(array.get_scalar_f32().is_err());
        assert_eq!(array.get_values_f32(), vec![123., 123., 123.]);
    }

    #[test]
    fn test_elementwise_unary_f32_scalar() {
        let hw = RefCell::new(CpuHardware::new());
        let x = Array::scalar_f32(&hw, 123.);

        let y = x.elementwise_neg_f32();
        assert_eq!(y.shape, make_shape![]);
        assert!(ptr::eq(y.buffer.hardware(), &hw));
        assert_eq!(y.get_values_f32(), vec![-123.]);
    }

    #[test]
    fn test_elementwise_unary_f32_0() {
        let hw = RefCell::new(CpuHardware::new());
        let x = Array::constant_f32(&hw, make_shape![0], &[]).unwrap();

        let y = x.elementwise_neg_f32();
        assert_eq!(y.shape, make_shape![0]);
        assert!(ptr::eq(y.buffer.hardware(), &hw));
        assert_eq!(y.get_values_f32(), vec![]);
    }

    #[test]
    fn test_elementwise_unary_f32_n() {
        let hw = RefCell::new(CpuHardware::new());
        let x = Array::constant_f32(&hw, make_shape![3], &[123., 456., 789.]).unwrap();

        let y = x.elementwise_neg_f32();
        assert_eq!(y.shape, make_shape![3]);
        assert!(ptr::eq(y.buffer.hardware(), &hw));
        assert_eq!(y.get_values_f32(), vec![-123., -456., -789.]);
    }

    #[test]
    fn test_elementwise_binary_f32_scalar_scalar() {
        let hw = RefCell::new(CpuHardware::new());
        let a = Array::scalar_f32(&hw, 111.);
        let b = Array::scalar_f32(&hw, 222.);

        let y = a.elementwise_add_f32(&b).unwrap();
        assert_eq!(y.shape, make_shape![]);
        assert!(ptr::eq(y.buffer.hardware(), &hw));
        assert_eq!(y.get_values_f32(), vec![333.]);

        let y = a.elementwise_sub_f32(&b).unwrap();
        assert_eq!(y.shape, make_shape![]);
        assert!(ptr::eq(y.buffer.hardware(), &hw));
        assert_eq!(y.get_values_f32(), vec![-111.]);

        let y = a.elementwise_mul_f32(&b).unwrap();
        assert_eq!(y.shape, make_shape![]);
        assert!(ptr::eq(y.buffer.hardware(), &hw));
        assert_eq!(y.get_values_f32(), vec![24642.]);

        let y = a.elementwise_div_f32(&b).unwrap();
        assert_eq!(y.shape, make_shape![]);
        assert!(ptr::eq(y.buffer.hardware(), &hw));
        assert_eq!(y.get_values_f32(), vec![0.5]);
    }

    #[test]
    fn test_elementwise_binary_f32_scalar_self() {
        let hw = RefCell::new(CpuHardware::new());
        let a = Array::scalar_f32(&hw, 111.);

        let y = a.elementwise_add_f32(&a).unwrap();
        assert_eq!(y.shape, make_shape![]);
        assert!(ptr::eq(y.buffer.hardware(), &hw));
        assert_eq!(y.get_values_f32(), vec![222.]);

        let y = a.elementwise_sub_f32(&a).unwrap();
        assert_eq!(y.shape, make_shape![]);
        assert!(ptr::eq(y.buffer.hardware(), &hw));
        assert_eq!(y.get_values_f32(), vec![0.]);

        let y = a.elementwise_mul_f32(&a).unwrap();
        assert_eq!(y.shape, make_shape![]);
        assert!(ptr::eq(y.buffer.hardware(), &hw));
        assert_eq!(y.get_values_f32(), vec![12321.]);

        let y = a.elementwise_div_f32(&a).unwrap();
        assert_eq!(y.shape, make_shape![]);
        assert!(ptr::eq(y.buffer.hardware(), &hw));
        assert_eq!(y.get_values_f32(), vec![1.]);
    }

    #[test]
    fn test_elementwise_binary_f32_0_0() {
        let hw = RefCell::new(CpuHardware::new());
        let a = Array::constant_f32(&hw, make_shape![0], &[]).unwrap();
        let b = Array::constant_f32(&hw, make_shape![0], &[]).unwrap();

        let y = a.elementwise_add_f32(&b).unwrap();
        assert_eq!(y.shape, make_shape![0]);
        assert!(ptr::eq(y.buffer.hardware(), &hw));
        assert_eq!(y.get_values_f32(), vec![]);

        let y = a.elementwise_sub_f32(&b).unwrap();
        assert_eq!(y.shape, make_shape![0]);
        assert!(ptr::eq(y.buffer.hardware(), &hw));
        assert_eq!(y.get_values_f32(), vec![]);

        let y = a.elementwise_mul_f32(&b).unwrap();
        assert_eq!(y.shape, make_shape![0]);
        assert!(ptr::eq(y.buffer.hardware(), &hw));
        assert_eq!(y.get_values_f32(), vec![]);

        let y = a.elementwise_div_f32(&b).unwrap();
        assert_eq!(y.shape, make_shape![0]);
        assert!(ptr::eq(y.buffer.hardware(), &hw));
        assert_eq!(y.get_values_f32(), vec![]);
    }

    #[test]
    fn test_elementwise_binary_f32_0_self() {
        let hw = RefCell::new(CpuHardware::new());
        let a = Array::constant_f32(&hw, make_shape![0], &[]).unwrap();

        let y = a.elementwise_add_f32(&a).unwrap();
        assert_eq!(y.shape, make_shape![0]);
        assert!(ptr::eq(y.buffer.hardware(), &hw));
        assert_eq!(y.get_values_f32(), vec![]);

        let y = a.elementwise_sub_f32(&a).unwrap();
        assert_eq!(y.shape, make_shape![0]);
        assert!(ptr::eq(y.buffer.hardware(), &hw));
        assert_eq!(y.get_values_f32(), vec![]);

        let y = a.elementwise_mul_f32(&a).unwrap();
        assert_eq!(y.shape, make_shape![0]);
        assert!(ptr::eq(y.buffer.hardware(), &hw));
        assert_eq!(y.get_values_f32(), vec![]);

        let y = a.elementwise_div_f32(&a).unwrap();
        assert_eq!(y.shape, make_shape![0]);
        assert!(ptr::eq(y.buffer.hardware(), &hw));
        assert_eq!(y.get_values_f32(), vec![]);
    }

    #[test]
    fn test_elementwise_binary_f32_n_n() {
        let hw = RefCell::new(CpuHardware::new());
        let a = Array::constant_f32(&hw, make_shape![3], &[111., 222., 333.]).unwrap();
        let b = Array::constant_f32(&hw, make_shape![3], &[444., 555., 666.]).unwrap();

        let y = a.elementwise_add_f32(&b).unwrap();
        assert_eq!(y.shape, make_shape![3]);
        assert!(ptr::eq(y.buffer.hardware(), &hw));
        assert_eq!(y.get_values_f32(), vec![555., 777., 999.]);

        let y = a.elementwise_sub_f32(&b).unwrap();
        assert_eq!(y.shape, make_shape![3]);
        assert!(ptr::eq(y.buffer.hardware(), &hw));
        assert_eq!(y.get_values_f32(), vec![-333., -333., -333.]);

        let y = a.elementwise_mul_f32(&b).unwrap();
        assert_eq!(y.shape, make_shape![3]);
        assert!(ptr::eq(y.buffer.hardware(), &hw));
        assert_eq!(y.get_values_f32(), vec![49284., 123210., 221778.]);

        let y = a.elementwise_div_f32(&b).unwrap();
        assert_eq!(y.shape, make_shape![3]);
        assert!(ptr::eq(y.buffer.hardware(), &hw));
        // Attempting exact matching even for 0.4.
        assert_eq!(y.get_values_f32(), vec![0.25, 0.4, 0.5]);
    }

    #[test]
    fn test_elementwise_binary_f32_n_self() {
        let hw = RefCell::new(CpuHardware::new());
        let a = Array::constant_f32(&hw, make_shape![3], &[111., 222., 333.]).unwrap();

        let y = a.elementwise_add_f32(&a).unwrap();
        assert_eq!(y.shape, make_shape![3]);
        assert!(ptr::eq(y.buffer.hardware(), &hw));
        assert_eq!(y.get_values_f32(), vec![222., 444., 666.]);

        let y = a.elementwise_sub_f32(&a).unwrap();
        assert_eq!(y.shape, make_shape![3]);
        assert!(ptr::eq(y.buffer.hardware(), &hw));
        assert_eq!(y.get_values_f32(), vec![0., 0., 0.]);

        let y = a.elementwise_mul_f32(&a).unwrap();
        assert_eq!(y.shape, make_shape![3]);
        assert!(ptr::eq(y.buffer.hardware(), &hw));
        assert_eq!(y.get_values_f32(), vec![12321., 49284., 110889.]);

        let y = a.elementwise_div_f32(&a).unwrap();
        assert_eq!(y.shape, make_shape![3]);
        assert!(ptr::eq(y.buffer.hardware(), &hw));
        assert_eq!(y.get_values_f32(), vec![1., 1., 1.]);
    }

    #[test]
    fn test_elementwise_binary_f32_scalar_0() {
        let hw = RefCell::new(CpuHardware::new());
        let a = Array::scalar_f32(&hw, 111.);
        let b = Array::constant_f32(&hw, make_shape![0], &[]).unwrap();

        assert!(a.elementwise_add_f32(&b).is_err());
        assert!(a.elementwise_sub_f32(&b).is_err());
        assert!(a.elementwise_mul_f32(&b).is_err());
        assert!(a.elementwise_div_f32(&b).is_err());

        assert!(b.elementwise_add_f32(&a).is_err());
        assert!(b.elementwise_sub_f32(&a).is_err());
        assert!(b.elementwise_mul_f32(&a).is_err());
        assert!(b.elementwise_div_f32(&a).is_err());
    }

    #[test]
    fn test_elementwise_binary_f32_scalar_1() {
        let hw = RefCell::new(CpuHardware::new());
        let a = Array::scalar_f32(&hw, 111.);
        let b = Array::constant_f32(&hw, make_shape![1], &[444.]).unwrap();

        assert!(a.elementwise_add_f32(&b).is_err());
        assert!(a.elementwise_sub_f32(&b).is_err());
        assert!(a.elementwise_mul_f32(&b).is_err());
        assert!(a.elementwise_div_f32(&b).is_err());

        assert!(b.elementwise_add_f32(&a).is_err());
        assert!(b.elementwise_sub_f32(&a).is_err());
        assert!(b.elementwise_mul_f32(&a).is_err());
        assert!(b.elementwise_div_f32(&a).is_err());
    }

    #[test]
    fn test_elementwise_binary_f32_scalar_n() {
        let hw = RefCell::new(CpuHardware::new());
        let a = Array::scalar_f32(&hw, 111.);
        let b = Array::constant_f32(&hw, make_shape![3], &[444., 555., 666.]).unwrap();

        assert!(a.elementwise_add_f32(&b).is_err());
        assert!(a.elementwise_sub_f32(&b).is_err());
        assert!(a.elementwise_mul_f32(&b).is_err());
        assert!(a.elementwise_div_f32(&b).is_err());

        assert!(b.elementwise_add_f32(&a).is_err());
        assert!(b.elementwise_sub_f32(&a).is_err());
        assert!(b.elementwise_mul_f32(&a).is_err());
        assert!(b.elementwise_div_f32(&a).is_err());
    }

    #[test]
    fn test_elementwise_binary_f32_0_1() {
        let hw = RefCell::new(CpuHardware::new());
        let a = Array::constant_f32(&hw, make_shape![0], &[]).unwrap();
        let b = Array::constant_f32(&hw, make_shape![1], &[111.]).unwrap();

        assert!(a.elementwise_add_f32(&b).is_err());
        assert!(a.elementwise_sub_f32(&b).is_err());
        assert!(a.elementwise_mul_f32(&b).is_err());
        assert!(a.elementwise_div_f32(&b).is_err());

        assert!(b.elementwise_add_f32(&a).is_err());
        assert!(b.elementwise_sub_f32(&a).is_err());
        assert!(b.elementwise_mul_f32(&a).is_err());
        assert!(b.elementwise_div_f32(&a).is_err());
    }

    #[test]
    fn test_elementwise_binary_f32_0_n() {
        let hw = RefCell::new(CpuHardware::new());
        let a = Array::constant_f32(&hw, make_shape![0], &[]).unwrap();
        let b = Array::constant_f32(&hw, make_shape![3], &[111., 222., 333.]).unwrap();

        assert!(a.elementwise_add_f32(&b).is_err());
        assert!(a.elementwise_sub_f32(&b).is_err());
        assert!(a.elementwise_mul_f32(&b).is_err());
        assert!(a.elementwise_div_f32(&b).is_err());

        assert!(b.elementwise_add_f32(&a).is_err());
        assert!(b.elementwise_sub_f32(&a).is_err());
        assert!(b.elementwise_mul_f32(&a).is_err());
        assert!(b.elementwise_div_f32(&a).is_err());
    }

    #[test]
    fn test_elementwise_binary_f32_1_n() {
        let hw = RefCell::new(CpuHardware::new());
        let a = Array::constant_f32(&hw, make_shape![1], &[111.]).unwrap();
        let b = Array::constant_f32(&hw, make_shape![3], &[444., 555., 666.]).unwrap();

        assert!(a.elementwise_add_f32(&b).is_err());
        assert!(a.elementwise_sub_f32(&b).is_err());
        assert!(a.elementwise_mul_f32(&b).is_err());
        assert!(a.elementwise_div_f32(&b).is_err());

        assert!(b.elementwise_add_f32(&a).is_err());
        assert!(b.elementwise_sub_f32(&a).is_err());
        assert!(b.elementwise_mul_f32(&a).is_err());
        assert!(b.elementwise_div_f32(&a).is_err());
    }

    #[test]
    fn test_elementwise_binary_f32_colocation() {
        let hw1 = RefCell::new(CpuHardware::new());
        let hw2 = RefCell::new(CpuHardware::new());
        let a = Array::scalar_f32(&hw1, 123.);
        let b = Array::scalar_f32(&hw2, 123.);

        assert!(a.elementwise_add_f32(&b).is_err());
        assert!(a.elementwise_sub_f32(&b).is_err());
        assert!(a.elementwise_mul_f32(&b).is_err());
        assert!(a.elementwise_div_f32(&b).is_err());

        assert!(b.elementwise_add_f32(&a).is_err());
        assert!(b.elementwise_sub_f32(&a).is_err());
        assert!(b.elementwise_mul_f32(&a).is_err());
        assert!(b.elementwise_div_f32(&a).is_err());
    }

    #[test]
    fn test_clone_scalar() {
        let hw = RefCell::new(CpuHardware::new());
        let x = Array::scalar_f32(&hw, 123.);
        let y = x.clone();
        assert_eq!(y.shape, make_shape![]);
        assert!(ptr::eq(y.buffer.hardware(), &hw));
        assert_eq!(y.get_values_f32(), vec![123.]);
    }

    #[test]
    fn test_clone_0() {
        let hw = RefCell::new(CpuHardware::new());
        let x = Array::constant_f32(&hw, make_shape![0], &[]).unwrap();
        let y = x.clone();
        assert_eq!(y.shape, make_shape![0]);
        assert!(ptr::eq(y.buffer.hardware(), &hw));
        assert_eq!(y.get_values_f32(), vec![]);
    }

    #[test]
    fn test_clone_n() {
        let hw = RefCell::new(CpuHardware::new());
        let x = Array::constant_f32(&hw, make_shape![3], &[123., 456., 789.]).unwrap();
        let y = x.clone();
        assert_eq!(y.shape, make_shape![3]);
        assert!(ptr::eq(y.buffer.hardware(), &hw));
        assert_eq!(y.get_values_f32(), vec![123., 456., 789.]);
    }
}

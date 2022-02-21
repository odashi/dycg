use crate::error::Error;
use crate::hardware::Hardware;
use crate::result::Result;
use std::cell::RefCell;
use std::ptr;

/// RAII object for hardware-specific memory.
///
/// This struct wraps around the raw handle returned by `Hardware`, and owns it during its lifetime.
/// At `drop()` the owned handle is released using associated `Hardware`.
///
/// This object can only be alive during the lifetime of the specified hardware.
pub struct Buffer<'hw> {
    /// Reference to the hardware that `pointer` manages.
    hardware: &'hw RefCell<dyn Hardware>,

    /// Size in bytes of the storage.
    size: usize,

    /// Handle of the device-specific storage.
    handle: *mut u8,
}

impl<'hw> Buffer<'hw> {
    /// Creates a new `Buffer` object without initialization.
    ///
    /// # Arguments
    ///
    /// * `hardware` - `HardwareMutex` to allocate the handle.
    /// * `size` - Size in bytes of the allocated memory.
    ///
    /// # Returns
    ///
    /// A new `Buffer` object.
    ///
    /// # Safety
    ///
    /// This function does not initialize the data on the allocated memory, and users are
    /// responsible to initialize the memory immediately by themselves.
    /// Using this object without explicit initialization causes undefined behavior.
    pub unsafe fn raw(hardware: &'hw RefCell<dyn Hardware>, size: usize) -> Self {
        Self {
            hardware,
            size,
            handle: hardware.borrow_mut().allocate_memory(size),
        }
    }

    /// Creates a new `Buffer` object on the same hardware of `other` without initialization.
    ///
    /// # Arguments
    ///
    /// * `other` - A `Buffer` object on the desired hardware.
    /// * `size` - Size in bytes of the allocated memory.
    ///
    /// # Returns
    ///
    /// A new `Buffer` object.
    ///
    /// # Safety
    ///
    /// This function does not initialize the data on the allocated memory, and users are
    /// responsible to initialize the memory immediately by themselves.
    /// Using this object without explicit initialization causes undefined behavior.
    pub unsafe fn raw_colocated(other: &Self, size: usize) -> Self {
        Self::raw(other.hardware, size)
    }

    /// Returns the hardware to manage owned memory.
    ///
    /// # Returns
    ///
    /// A Reference to the wrapped `Hardware` object.
    pub fn hardware(&self) -> &'hw RefCell<dyn Hardware> {
        self.hardware
    }

    /// Returns the size of the allocated memory.
    ///
    /// # Returns
    ///
    /// The size of the allocated memory.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the const handle owned by this buffer.
    ///
    /// # Returns
    ///
    /// Owned handle as a const pointer.
    ///
    /// # Safety
    ///
    /// This function returns a raw pointer of the inner memory or a handle of the associated
    /// hardware-specific object.
    /// The returned value can not be used without knowing the associated hardware.
    pub unsafe fn as_handle(&self) -> *const u8 {
        self.handle
    }

    /// Returns the mutable handle owned by this buffer.
    ///
    /// # Returns
    ///
    /// Owned handle as a mutable pointer.
    ///
    /// # Safety
    ///
    /// This function returns a raw pointer of the inner memory or a handle of the associated
    /// device-specific object.
    /// The returned value can not be used without knowing the associated hardware.
    pub unsafe fn as_mut_handle(&mut self) -> *mut u8 {
        self.handle
    }

    /// Checks if the both buffers are colocated on the same hardware.
    ///
    /// # Arguments
    ///
    /// * `other` - A `Buffer` object to check colocation.
    ///
    /// # Returns
    ///
    /// * `true` - The both buffers are colocated on the same hardware.
    /// * `false` - Otherwise.
    pub fn is_colocated(&self, other: &Self) -> bool {
        ptr::eq(self.hardware, other.hardware)
    }

    /// Checks if the both buffers are colocated on the same hardware.
    ///
    /// # Arguments
    ///
    /// * `other` - A `Buffer` object to check colocation.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - The both buffers are colocated on the same hardware.
    /// * `Err(Error)` - Otherwise.
    pub fn check_colocated(&self, other: &Self) -> Result<()> {
        self.is_colocated(other).then(|| ()).ok_or_else(|| {
            Error::InvalidHardware(format!(
                "Buffers are not colocated on the same hardware. self: {:p}, other: {:p}",
                self.hardware, other.hardware,
            ))
        })
    }
}

impl<'hw> Drop for Buffer<'hw> {
    fn drop(&mut self) {
        unsafe {
            self.hardware
                .borrow_mut()
                .deallocate_memory(self.handle, self.size);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::buffer::Buffer;
    use crate::hardware::cpu::CpuHardware;
    use std::cell::RefCell;
    use std::ptr;

    #[test]
    fn test_raw() {
        let hw1 = RefCell::new(CpuHardware::new());
        let hw2 = RefCell::new(CpuHardware::new());
        let nullptr = ptr::null::<u8>();
        unsafe {
            let buf1 = Buffer::raw(&hw1, 1);
            assert!(ptr::eq(buf1.hardware, &hw1));
            assert_eq!(buf1.size, 1);
            assert!(!ptr::eq(buf1.handle, nullptr));

            let buf2 = Buffer::raw(&hw1, 2);
            assert!(ptr::eq(buf2.hardware, &hw1));
            assert_eq!(buf2.size, 2);
            assert!(!ptr::eq(buf2.handle, nullptr));
            assert!(!ptr::eq(buf2.handle, buf1.handle));

            let buf3 = Buffer::raw(&hw2, 3);
            assert!(ptr::eq(buf3.hardware, &hw2));
            assert_eq!(buf3.size, 3);
            assert!(!ptr::eq(buf3.handle, nullptr));
            assert!(!ptr::eq(buf3.handle, buf1.handle));
            assert!(!ptr::eq(buf3.handle, buf2.handle));
        }
    }

    #[test]
    fn test_raw_zero() {
        let hw = RefCell::new(CpuHardware::new());
        unsafe {
            let buf = Buffer::raw(&hw, 0);
            assert!(ptr::eq(buf.hardware, &hw));
            assert_eq!(buf.size, 0);
            // We don't care about the pointer value of zero-length memory.
        }
    }

    #[test]
    fn test_raw_colocated() {
        let hw = RefCell::new(CpuHardware::new());
        unsafe {
            let buf1 = Buffer::raw(&hw, 1);
            let buf2 = Buffer::raw_colocated(&buf1, 2);
            assert!(ptr::eq(buf2.hardware, &hw));
            assert_eq!(buf2.size, 2);
            assert!(!ptr::eq(buf2.handle, buf1.handle));
        }
    }

    #[test]
    fn test_hardware() {
        let hw = RefCell::new(CpuHardware::new());
        unsafe {
            let buf = Buffer::raw(&hw, 1);
            assert!(ptr::eq(buf.hardware(), &hw));
        }
    }

    #[test]
    fn test_size() {
        let hw = RefCell::new(CpuHardware::new());
        unsafe {
            let buf = Buffer::raw(&hw, 123);
            assert_eq!(buf.size(), 123);
        }
    }

    #[test]
    fn test_as_handle() {
        let hw = RefCell::new(CpuHardware::new());
        unsafe {
            let buf = Buffer::raw(&hw, 1);
            assert!(ptr::eq(buf.as_handle(), buf.handle));
        }
    }

    #[test]
    fn test_as_mut_handle() {
        let hw = RefCell::new(CpuHardware::new());
        unsafe {
            let mut buf = Buffer::raw(&hw, 1);
            assert!(ptr::eq(buf.as_mut_handle(), buf.handle));
        }
    }

    #[test]
    fn test_is_colocated() {
        let hw1 = RefCell::new(CpuHardware::new());
        let hw2 = RefCell::new(CpuHardware::new());
        unsafe {
            let buf1 = Buffer::raw(&hw1, 1);
            let buf2 = Buffer::raw(&hw1, 1);
            let buf3 = Buffer::raw(&hw2, 1);
            assert!(buf1.is_colocated(&buf1));
            assert!(buf1.is_colocated(&buf2));
            assert!(!buf1.is_colocated(&buf3));
            assert!(buf2.is_colocated(&buf1));
            assert!(buf2.is_colocated(&buf2));
            assert!(!buf2.is_colocated(&buf3));
            assert!(!buf3.is_colocated(&buf1));
            assert!(!buf3.is_colocated(&buf2));
            assert!(buf3.is_colocated(&buf3));
        }
    }

    #[test]
    fn test_check_colocated() {
        let hw1 = RefCell::new(CpuHardware::new());
        let hw2 = RefCell::new(CpuHardware::new());
        unsafe {
            let buf1 = Buffer::raw(&hw1, 1);
            let buf2 = Buffer::raw(&hw1, 1);
            let buf3 = Buffer::raw(&hw2, 1);
            assert!(buf1.check_colocated(&buf1).is_ok());
            assert!(buf1.check_colocated(&buf2).is_ok());
            assert!(buf1.check_colocated(&buf3).is_err());
            assert!(buf2.check_colocated(&buf1).is_ok());
            assert!(buf2.check_colocated(&buf2).is_ok());
            assert!(buf2.check_colocated(&buf3).is_err());
            assert!(buf3.check_colocated(&buf1).is_err());
            assert!(buf3.check_colocated(&buf2).is_err());
            assert!(buf3.check_colocated(&buf3).is_ok());
        }
    }
}

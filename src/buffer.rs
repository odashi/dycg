use crate::error::Error;
use crate::hardware::HardwareMutex;
use crate::result::Result;
use std::ptr;

/// RAII object for hardware-specific memory.
///
/// This struct wraps around the raw handle returned by `Hardware`, and owns it during its lifetime.
/// At `drop()` the owned handle is released using associated `Hardware`.
///
/// This object can only be alive during the lifetime of the specified hardware.
pub(crate) struct Buffer<'hw> {
    /// Reference to the hardware that `pointer` manages.
    hardware: &'hw HardwareMutex,

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
    pub(crate) unsafe fn raw(hardware: &'hw HardwareMutex, size: usize) -> Self {
        // Panics immediately when mutex poisoning happened.
        Self {
            hardware,
            size,
            handle: hardware.lock().unwrap().allocate_memory(size),
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
    pub(crate) unsafe fn raw_colocated(other: &Buffer<'hw>, size: usize) -> Self {
        Self::raw(other.hardware, size)
    }

    /// Returns the hardware to manage owned memory.
    ///
    /// # Returns
    ///
    /// A Reference to the wrapped `Hardware` object.
    pub(crate) fn hardware(&self) -> &'hw HardwareMutex {
        self.hardware
    }

    /// Returns the size of the owned memory.
    ///
    /// # Returns
    ///
    /// The size of the owned memory.
    pub(crate) fn size(&self) -> usize {
        self.size
    }

    /// Returns the const handle owned by this buffer.
    ///
    /// # Returns
    ///
    /// Owned handle as a const pointer.
    pub(crate) unsafe fn as_handle(&self) -> *const u8 {
        self.handle
    }

    /// Returns the mutable handle owned by this buffer.
    ///
    /// # Returns
    ///
    /// Owned handle as a mutable pointer.
    pub(crate) unsafe fn as_mut_handle(&mut self) -> *mut u8 {
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
    pub(crate) fn is_colocated(&self, other: &Buffer) -> bool {
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
    pub(crate) fn check_colocated(&self, other: &Buffer) -> Result<()> {
        self.is_colocated(other)
            .then(|| ())
            .ok_or(Error::InvalidHardware((|| {
                format!(
                    "Buffers are not colocated on the same hardware. self: {:p}, other: {:p}",
                    self.hardware, other.hardware,
                )
            })()))
    }
}

impl<'hw> Drop for Buffer<'hw> {
    fn drop(&mut self) {
        unsafe {
            // Panics immediately when mutex poisoning happened.
            self.hardware
                .lock()
                .unwrap()
                .deallocate_memory(self.handle, self.size);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::buffer::Buffer;
    use crate::hardware::{cpu::CpuHardware, HardwareMutex};
    use std::ptr;

    /// Helper function to create mutex-guarded CpuHardwre.
    fn make_hardware() -> HardwareMutex {
        HardwareMutex::new(Box::new(CpuHardware::new("test")))
    }

    #[test]
    fn test_raw() {
        let hw1 = make_hardware();
        let hw2 = make_hardware();
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
        let hw = make_hardware();
        unsafe {
            let buf = Buffer::raw(&hw, 0);
            assert!(ptr::eq(buf.hardware, &hw));
            assert_eq!(buf.size, 0);
            // We don't care about the pointer value of zero-length memory.
        }
    }

    #[test]
    fn test_raw_colocated() {
        let hw = make_hardware();
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
        let hw = make_hardware();
        unsafe {
            let buf = Buffer::raw(&hw, 1);
            assert!(ptr::eq(buf.hardware(), &hw));
        }
    }

    #[test]
    fn test_size() {
        let hw = make_hardware();
        unsafe {
            let buf = Buffer::raw(&hw, 123);
            assert_eq!(buf.size(), 123);
        }
    }

    #[test]
    fn test_as_handle() {
        let hw = make_hardware();
        unsafe {
            let buf = Buffer::raw(&hw, 1);
            assert!(ptr::eq(buf.as_handle(), buf.handle));
        }
    }

    #[test]
    fn test_as_mut_handle() {
        let hw = make_hardware();
        unsafe {
            let mut buf = Buffer::raw(&hw, 1);
            assert!(ptr::eq(buf.as_mut_handle(), buf.handle));
        }
    }

    #[test]
    fn test_is_colocated() {
        let hw1 = make_hardware();
        let hw2 = make_hardware();
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
        let hw1 = make_hardware();
        let hw2 = make_hardware();
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

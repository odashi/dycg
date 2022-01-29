use crate::error::Error;
use crate::hardware::Hardware;
use crate::result::Result;
use std::ptr;
use std::sync::Mutex;

/// RAII object for hardware-specific memory.
///
/// This struct wraps around the raw handle returned by `Hardware`, and owns it during its lifetime.
/// At `drop()` the owned handle is released using associated `Hardware`.
///
/// This object can only be alive during the lifetime of the specified hardware.
pub(crate) struct Buffer<'hw> {
    /// Reference to the hardware that `pointer` manages.
    hardware: &'hw Mutex<Box<dyn Hardware>>,

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
    /// * `hardware` - `Hardware` to allocate the handle.
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
    pub(crate) unsafe fn raw(hardware: &'hw Mutex<Box<dyn Hardware>>, size: usize) -> Self {
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
    pub(crate) fn hardware(&self) -> &'hw Mutex<Box<dyn Hardware>> {
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

use crate::hardware::Hardware;
use std::sync::Mutex;

/// RAII object for hardware-specific memory.
///
/// This struct wraps around the raw handle returned by `Hardware`, and owns it during its lifetime.
/// At `drop()` the owned handle is released using associated `Hardware`.
pub(crate) struct Buffer<'a> {
    /// Reference to the hardware that `pointer` manages.
    hardware: &'a Mutex<Box<dyn Hardware>>,

    /// Size in bytes of the storage.
    size: usize,

    /// Handle of the device-specific storage.
    handle: *mut u8,
}

impl<'a> Buffer<'a> {
    /// Creates a new `Buffer` object.
    ///
    /// # Arguments
    ///
    /// * `hardware` - `Hardware` to allocate the handle.
    /// * `size` - Size in bytes of the allocated memory.
    ///
    /// # Returns
    ///
    /// A new `Buffer` object owning allocated memory.
    pub(crate) fn new(hardware: &'a Mutex<Box<dyn Hardware>>, size: usize) -> Self {
        Self {
            hardware,
            size,
            handle: unsafe {
                // Panics immediately when mutex poisoning happened.
                hardware.lock().unwrap().allocate_memory(size)
            },
        }
    }

    /// Returns the hardware to manage owned memory.
    ///
    /// # Returns
    ///
    /// A Reference to the wrapped `Hardware` object.
    pub(crate) fn hardware(&self) -> &'a Mutex<Box<dyn Hardware>> {
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
    /// # Returns:
    ///
    /// Owned handle as a const pointer.
    pub(crate) unsafe fn as_handle(&self) -> *const u8 {
        self.handle
    }

    /// Returns the mutable handle owned by this buffer.
    ///
    /// # Returns:
    ///
    /// Owned handle as a mutable pointer.
    pub(crate) unsafe fn as_handle_mut(&mut self) -> *mut u8 {
        self.handle
    }
}

impl<'a> Drop for Buffer<'a> {
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

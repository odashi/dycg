use crate::backend::Backend;
use std::sync::Mutex;

/// Device-specific memory.
///
/// This struct wraps around the raw handle returned by `Backend`, and owns it during its lifetime.
/// At `drop()` the owned handle is released using associated `Backend.
pub(crate) struct Buffer<'a> {
    /// Reference to the backend that `pointer` manages.
    backend: &'a Mutex<Box<dyn Backend>>,

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
    /// * `backend` - `Backend` to allocate the handle.
    /// * `size` - Size in bytes of the allocated memory.
    ///
    /// # Returns
    ///
    /// A new `Buffer` object owning allocated memory.
    pub(crate) fn new(backend: &'a Mutex<Box<dyn Backend>>, size: usize) -> Self {
        Self {
            backend,
            size,
            handle: unsafe {
                // Panics immediately when mutex poisoning happened.
                backend.lock().unwrap().get_memory(size)
            },
        }
    }

    /// Returns the backend to manage owned memory.
    ///
    /// # Returns
    ///
    /// A Reference to the wrapped `Backend` object.
    pub(crate) fn backend(&self) -> &'a Mutex<Box<dyn Backend>> {
        self.backend
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
            self.backend
                .lock()
                .unwrap()
                .release_memory(self.handle, self.size);
        }
    }
}

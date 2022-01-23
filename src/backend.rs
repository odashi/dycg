use once_cell::sync::OnceCell;
use std::alloc;
use std::ops;
use std::sync::Mutex;

/// Default memory alignment for allocating buffers.
const DEFAULT_MEMORY_ALIGNMENT: usize = 8;

/// Trait for computing backends.
pub(crate) trait Backend {
    /// Allocates a new memory with requested size and returns its handle.
    ///
    /// This function may abort when memory allocation failed.
    ///
    /// # Arguments
    ///
    /// * `size` - Size in bytes of the memory.
    ///
    /// # Returns
    ///
    /// `Handle` of the created memory.
    ///
    unsafe fn get_memory(&mut self, size: usize) -> *mut u8;

    /// Releases given buffer.
    ///
    /// After calling this function, `memory` must not be used because it no longer points to any
    /// valid data.
    ///
    /// # Arguments
    ///
    /// * `handle` - `Handle` object to release. This value must be one returned by `get_memory` of
    ///   the same backend.
    /// * `size` - Size in bytes of the allocated memory. This value must be equal to that specified
    ///   at corresponding `get_memory` call.
    unsafe fn release_memory(&mut self, handle: *mut u8, size: usize);

    /// Copies data from a host memory to a backend memory.
    ///
    /// # Arguments
    ///
    /// * `src` - Source host memory.
    /// * `dest` - Target backend memory.
    /// * `size` - Size in bytes to copy.
    ///
    /// # Requirements
    ///
    /// Both `src` and `dest` owns enough amount of memory to store data with `size` bytes long.
    unsafe fn copy_host_to_backend(&mut self, src: *const u8, dest: *mut u8, size: usize);

    /// Copies data from a backend memory to a host memory.
    ///
    /// # Arguments
    ///
    /// * `src` - Source backend memory.
    /// * `dest` - Target host memory.
    /// * `size` - Size in bytes to copy.
    ///
    /// # Requirements
    ///
    /// Both `src` and `dest` own enough amount of memory to store data with `size` bytes long.
    unsafe fn copy_backend_to_host(&mut self, src: *const u8, dest: *mut u8, size: usize);

    /// Performs elementwise add operation.
    ///
    /// # Arguments
    ///
    /// * `lhs` - Backend memory for left-hand side argument.
    /// * `rhs` - Backend memory for right-hand side argument.
    /// * `dest` - Backend memory for destination.
    /// * `num_elements` - Number of elements to be calculated.
    ///
    /// # Requirements
    ///
    /// `lhs`, `rhs`, and `dest` own enough amount of memory to store data with `num_elements`
    /// elements of the value type.
    unsafe fn elementwise_add_f32(
        &mut self,
        lhs: *const u8,
        rhs: *const u8,
        dest: *mut u8,
        size: usize,
    );

    /// Performs elementwise subtract operation.
    ///
    /// # Arguments
    ///
    /// * `lhs` - Backend memory for left-hand side argument.
    /// * `rhs` - Backend memory for right-hand side argument.
    /// * `dest` - Backend memory for destination.
    /// * `num_elements` - Number of elements to be calculated.
    ///
    /// # Requirements
    ///
    /// `lhs`, `rhs`, and `dest` own enough amount of memory to store data with `num_elements`
    /// elements of the value type.
    unsafe fn elementwise_sub_f32(
        &mut self,
        lhs: *const u8,
        rhs: *const u8,
        dest: *mut u8,
        size: usize,
    );

    /// Performs elementwise multiply operation.
    ///
    /// # Arguments
    ///
    /// * `lhs` - Backend memory for left-hand side argument.
    /// * `rhs` - Backend memory for right-hand side argument.
    /// * `dest` - Backend memory for destination.
    /// * `num_elements` - Number of elements to be calculated.
    ///
    /// # Requirements
    ///
    /// `lhs`, `rhs`, and `dest` own enough amount of memory to store data with `num_elements`
    /// elements of the value type.
    unsafe fn elementwise_mul_f32(
        &mut self,
        lhs: *const u8,
        rhs: *const u8,
        dest: *mut u8,
        size: usize,
    );

    /// Performs elementwise divide operation.
    ///
    /// # Arguments
    ///
    /// * `lhs` - Backend memory for left-hand side argument.
    /// * `rhs` - Backend memory for right-hand side argument.
    /// * `dest` - Backend memory for destination.
    /// * `num_elements` - Number of elements to be calculated.
    ///
    /// # Requirements
    ///
    /// `lhs`, `rhs`, and `dest` own enough amount of memory to store data with `num_elements`
    /// elements of the value type.
    unsafe fn elementwise_div_f32(
        &mut self,
        lhs: *const u8,
        rhs: *const u8,
        dest: *mut u8,
        size: usize,
    );
}

/// Backend for computation on local CPUs.
///
/// Memories are allocated through `GlobalAlloc`.
pub struct CpuBackend;

impl CpuBackend {
    /// Creates a new `CpuBackend` object.
    ///
    /// # Returns
    ///
    /// A new `CpuBackend` object.
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl Backend for CpuBackend {
    unsafe fn get_memory(&mut self, size: usize) -> *mut u8 {
        let layout = alloc::Layout::from_size_align_unchecked(size, DEFAULT_MEMORY_ALIGNMENT);
        let handle = alloc::alloc(layout);
        if handle.is_null() {
            // Panics immediately when allocation error occurred.
            alloc::handle_alloc_error(layout);
        }
        println!(
            "Allocated a buffer: handle={:16x}, size={}",
            unsafe { handle as usize },
            size
        );
        handle
    }

    unsafe fn release_memory(&mut self, handle: *mut u8, size: usize) {
        println!(
            "Released a buffer: handle={:16x}, size={}",
            unsafe { handle as usize },
            size
        );
        alloc::dealloc(
            handle,
            alloc::Layout::from_size_align_unchecked(size, DEFAULT_MEMORY_ALIGNMENT),
        )
    }

    unsafe fn copy_host_to_backend(&mut self, src: *const u8, dest: *mut u8, size: usize) {
        std::ptr::copy(src, dest as *mut u8, size);
    }

    unsafe fn copy_backend_to_host(&mut self, src: *const u8, dest: *mut u8, size: usize) {
        std::ptr::copy(src as *const u8, dest, size)
    }

    unsafe fn elementwise_add_f32(
        &mut self,
        lhs: *const u8,
        rhs: *const u8,
        dest: *mut u8,
        size: usize,
    ) {
        let lhs = lhs as *const f32;
        let rhs = lhs as *const f32;
        let dest = lhs as *mut f32;
        for i in 0..size {
            *dest.add(i) = *lhs.add(i) + *rhs.add(i);
        }
    }

    unsafe fn elementwise_sub_f32(
        &mut self,
        lhs: *const u8,
        rhs: *const u8,
        dest: *mut u8,
        size: usize,
    ) {
        let lhs = lhs as *const f32;
        let rhs = lhs as *const f32;
        let dest = lhs as *mut f32;
        for i in 0..size {
            *dest.add(i) = *lhs.add(i) - *rhs.add(i);
        }
    }
    unsafe fn elementwise_mul_f32(
        &mut self,
        lhs: *const u8,
        rhs: *const u8,
        dest: *mut u8,
        size: usize,
    ) {
        let lhs = lhs as *const f32;
        let rhs = lhs as *const f32;
        let dest = lhs as *mut f32;
        for i in 0..size {
            *dest.add(i) = *lhs.add(i) * *rhs.add(i);
        }
    }
    unsafe fn elementwise_div_f32(
        &mut self,
        lhs: *const u8,
        rhs: *const u8,
        dest: *mut u8,
        size: usize,
    ) {
        let lhs = lhs as *const f32;
        let rhs = lhs as *const f32;
        let dest = lhs as *mut f32;
        for i in 0..size {
            *dest.add(i) = *lhs.add(i) / *rhs.add(i);
        }
    }
}

/// Returns the default backend.
///
/// Any arrays without explicit specification of backend falls back to use this backend.
pub(crate) fn get_default_backend() -> &'static Mutex<Box<dyn Backend>> {
    static SINGLETON: OnceCell<Mutex<Box<dyn Backend>>> = OnceCell::new();
    SINGLETON.get_or_init(|| Mutex::new(Box::new(CpuBackend::new())))
}

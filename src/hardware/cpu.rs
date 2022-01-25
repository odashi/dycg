use std::alloc;

/// Default memory alignment for allocating buffers.
const DEFAULT_MEMORY_ALIGNMENT: usize = 8;

/// Hardware for computation on local CPUs.
///
/// Memories on this hardware are identical with the usual host memory and are allocated through
/// `GlobalAlloc`.
pub struct Cpu {
    /// Name of this hardware.
    name: String,
}

impl Cpu {
    /// Creates a new `CpuHardware` object.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of this hardware.
    ///
    /// # Returns
    ///
    /// A new `CpuHardware` object.
    pub(crate) fn new(name: &str) -> Self {
        Self {
            name: String::from(name),
        }
    }
}

unsafe impl crate::hardware::Hardware for Cpu {
    fn name(&self) -> &str {
        &self.name
    }

    unsafe fn allocate_memory(&mut self, size: usize) -> *mut u8 {
        let layout = alloc::Layout::from_size_align_unchecked(size, DEFAULT_MEMORY_ALIGNMENT);
        let handle = alloc::alloc(layout);
        if handle.is_null() {
            // Panics immediately when allocation error occurred.
            alloc::handle_alloc_error(layout);
        }
        eprintln!("Allocated a buffer: handle={:p}, size={}", handle, size);
        handle
    }

    unsafe fn deallocate_memory(&mut self, handle: *mut u8, size: usize) {
        eprintln!("Released a buffer: handle={:p}, size={}", handle, size);
        alloc::dealloc(
            handle,
            alloc::Layout::from_size_align_unchecked(size, DEFAULT_MEMORY_ALIGNMENT),
        )
    }

    unsafe fn copy_host_to_hardware(&mut self, src: *const u8, dest: *mut u8, size: usize) {
        std::ptr::copy(src, dest as *mut u8, size);
    }

    unsafe fn copy_hardware_to_host(&mut self, src: *const u8, dest: *mut u8, size: usize) {
        std::ptr::copy(src as *const u8, dest, size)
    }

    unsafe fn fill_f32(&mut self, src: *mut u8, value: f32, num_elements: usize) {
        let src = src as *mut f32;
        for i in 0..num_elements {
            *src.add(i) = value;
        }
    }

    unsafe fn elementwise_add_f32(
        &mut self,
        lhs: *const u8,
        rhs: *const u8,
        dest: *mut u8,
        num_elements: usize,
    ) {
        let lhs = lhs as *const f32;
        let rhs = rhs as *const f32;
        let dest = dest as *mut f32;
        for i in 0..num_elements {
            *dest.add(i) = *lhs.add(i) + *rhs.add(i);
        }
    }

    unsafe fn elementwise_sub_f32(
        &mut self,
        lhs: *const u8,
        rhs: *const u8,
        dest: *mut u8,
        num_elements: usize,
    ) {
        let lhs = lhs as *const f32;
        let rhs = rhs as *const f32;
        let dest = dest as *mut f32;
        for i in 0..num_elements {
            *dest.add(i) = *lhs.add(i) - *rhs.add(i);
        }
    }
    unsafe fn elementwise_mul_f32(
        &mut self,
        lhs: *const u8,
        rhs: *const u8,
        dest: *mut u8,
        num_elements: usize,
    ) {
        let lhs = lhs as *const f32;
        let rhs = rhs as *const f32;
        let dest = dest as *mut f32;
        for i in 0..num_elements {
            *dest.add(i) = *lhs.add(i) * *rhs.add(i);
        }
    }
    unsafe fn elementwise_div_f32(
        &mut self,
        lhs: *const u8,
        rhs: *const u8,
        dest: *mut u8,
        num_elements: usize,
    ) {
        let lhs = lhs as *const f32;
        let rhs = rhs as *const f32;
        let dest = dest as *mut f32;
        for i in 0..num_elements {
            *dest.add(i) = *lhs.add(i) / *rhs.add(i);
        }
    }
}

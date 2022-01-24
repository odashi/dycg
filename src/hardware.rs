use once_cell::sync::OnceCell;
use std::alloc;
use std::sync::Mutex;

/// Default memory alignment for allocating buffers.
const DEFAULT_MEMORY_ALIGNMENT: usize = 8;

/// Trait for computing backends.
///
/// This trait provides the set of the lowest instructions that each computation backend are
/// required to implement. Higher abstraction for user-level computation is provided by `Array`.
///
/// As the real hardware lives longer than the programs, structs implementing this trait may be
/// installed as a static object.
/// They require implicit/explicit initialization procedure during the program startups.
pub(crate) unsafe trait Hardware {
    /// Allocates a new memory with at least the requested size and returns its handle.
    ///
    /// # Arguments
    ///
    /// * `size` - Size in bytes of the memory.
    ///
    /// # Returns
    ///
    /// `Handle` of the created memory.
    /// The handle value may or may not represent a real memory. For example, GPU hardwares may
    /// represent a virtual address representing a corresponding region on the VRAM.
    ///
    /// # Panics
    ///
    /// This function may panic when memory allocation failed for some reason and the implementation
    /// judged that the failure can not be recovered.
    ///
    /// # Safety
    ///
    /// The memory returned by this function may not be initialized. Users are responsible to
    /// initialize the memory immediately by themselves.
    unsafe fn allocate_memory(&mut self, size: usize) -> *mut u8;

    /// Releases given buffer.
    ///
    /// # Arguments
    ///
    /// * `handle` - `Handle` object to release. This value must be one returned by `get_memory` of
    ///   the same hardware.
    /// * `size` - Size in bytes of the allocated memory. This value must be equal to that specified
    ///   at corresponding `get_memory` call.
    ///
    /// # Safety
    ///
    /// After calling this function, `memory` must not be used because it no longer points to any
    /// valid data.
    unsafe fn deallocate_memory(&mut self, handle: *mut u8, size: usize);

    /// Copies data from a host memory to a hardware memory.
    ///
    /// # Arguments
    ///
    /// * `src` - Source host memory.
    /// * `dest` - Target hardware memory.
    /// * `size` - Size in bytes to copy.
    ///
    /// # Requirements
    ///
    /// Both `src` and `dest` owns enough amount of memory to store data with `size` bytes long.
    unsafe fn copy_host_to_hardware(&mut self, src: *const u8, dest: *mut u8, size: usize);

    /// Copies data from a hardware memory to a host memory.
    ///
    /// # Arguments
    ///
    /// * `src` - Source hardware memory.
    /// * `dest` - Target host memory.
    /// * `size` - Size in bytes to copy.
    ///
    /// # Requirements
    ///
    /// Both `src` and `dest` own enough amount of memory to store data with `size` bytes long.
    unsafe fn copy_hardware_to_host(&mut self, src: *const u8, dest: *mut u8, size: usize);

    /// Fills the memory with specified data.
    ///
    /// # Arguments
    ///
    /// * `src` - Hardware memory to be filled.
    /// * `value` - Value to fill.
    /// * `num_elements` - Number of elements to be filled.
    ///
    /// # Requirements
    ///
    /// `src` own enough amount of memory to store data with `num_elements` elements of the value
    /// type.
    unsafe fn fill_f32(&mut self, src: *mut u8, value: f32, num_elements: usize);

    /// Performs elementwise add operation.
    ///
    /// # Arguments
    ///
    /// * `lhs` - Hardware memory for left-hand side argument.
    /// * `rhs` - Hardware memory for right-hand side argument.
    /// * `dest` - Hardware memory for destination.
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
        num_elements: usize,
    );

    /// Performs elementwise subtract operation.
    ///
    /// # Arguments
    ///
    /// * `lhs` - Hardware memory for left-hand side argument.
    /// * `rhs` - Hardware memory for right-hand side argument.
    /// * `dest` - Hardware memory for destination.
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
        num_elements: usize,
    );

    /// Performs elementwise multiply operation.
    ///
    /// # Arguments
    ///
    /// * `lhs` - Hardware memory for left-hand side argument.
    /// * `rhs` - Hardware memory for right-hand side argument.
    /// * `dest` - Hardware memory for destination.
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
        num_elements: usize,
    );

    /// Performs elementwise divide operation.
    ///
    /// # Arguments
    ///
    /// * `lhs` - Hardware memory for left-hand side argument.
    /// * `rhs` - Hardware memory for right-hand side argument.
    /// * `dest` - Hardware memory for destination.
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
        num_elements: usize,
    );
}

/// Hardware for computation on local CPUs.
///
/// Memories on this hardware are identical with the usual host memory and are allocated through
/// `GlobalAlloc`.
pub struct CpuHardware;

impl CpuHardware {
    /// Creates a new `CpuHardware` object.
    ///
    /// # Returns
    ///
    /// A new `CpuHardware` object.
    fn new() -> Self {
        Self {}
    }
}

unsafe impl Hardware for CpuHardware {
    unsafe fn allocate_memory(&mut self, size: usize) -> *mut u8 {
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

    unsafe fn deallocate_memory(&mut self, handle: *mut u8, size: usize) {
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
        let rhs = lhs as *const f32;
        let dest = lhs as *mut f32;
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
        let rhs = lhs as *const f32;
        let dest = lhs as *mut f32;
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
        let rhs = lhs as *const f32;
        let dest = lhs as *mut f32;
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
        let rhs = lhs as *const f32;
        let dest = lhs as *mut f32;
        for i in 0..num_elements {
            *dest.add(i) = *lhs.add(i) / *rhs.add(i);
        }
    }
}

/// Returns the default hardware.
///
/// Any arrays without explicit specification of hardware falls back to use this hardware.
pub(crate) fn get_default_hardware() -> &'static Mutex<Box<dyn Hardware>> {
    static SINGLETON: OnceCell<Mutex<Box<dyn Hardware>>> = OnceCell::new();
    SINGLETON.get_or_init(|| Mutex::new(Box::new(CpuHardware::new())))
}

/// Trait for computing backends.
///
/// This trait provides the set of the lowest instructions that each computation backend are
/// required to implement. Higher abstraction for user-level computation is provided by `Array`.
///
/// As the real hardware lives longer than the programs, structs implementing this trait may be
/// installed as a static object.
/// They require implicit/explicit initialization procedure during the program startups.
///
/// # Safety
///
/// This trait treats unsafe memory blocks or similar hardware-specific objects.
pub unsafe trait Hardware {
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
    /// # Safety
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
    /// # Safety
    ///
    /// Both `src` and `dest` own enough amount of memory to store data with `size` bytes long.
    unsafe fn copy_hardware_to_host(&mut self, src: *const u8, dest: *mut u8, size: usize);

    /// Copies data between hardware memories.
    ///
    /// # Arguments
    ///
    /// * `src` - Source hardware memory.
    /// * `dest` - Target hardware memory.
    /// * `size` - Size in bytes to copy.
    ///
    /// # Safety
    ///
    /// Both `src` and `dest` own enough amount of memory to store data with `size` bytes long.
    unsafe fn copy_hardware_to_hardware(&mut self, src: *const u8, dest: *mut u8, size: usize);

    /// Fills the memory with specified data.
    ///
    /// # Arguments
    ///
    /// * `dest` - Hardware memory to be filled.
    /// * `value` - Value to fill.
    /// * `num_elements` - Number of elements to be filled.
    ///
    /// # Safety
    ///
    /// `src` own enough amount of memory to store data with `num_elements` elements of the value
    /// type.
    unsafe fn fill_f32(&mut self, dest: *mut u8, value: f32, num_elements: usize);

    /// Performs elementwise negation operation.
    ///
    /// # Arguments
    ///
    /// * `src` - Hardware memory for the source.
    /// * `dest` - Hardware memory for the destination.
    /// * `num_elements` - Number of elements on each memory.
    ///
    /// # Safety
    ///
    /// `src` and `dest` own enough amount of memory to store data with `num_elements` elements
    /// of the value type.
    unsafe fn elementwise_neg_f32(&mut self, src: *const u8, dest: *mut u8, num_elements: usize);

    /// Performs elementwise add operation.
    ///
    /// # Arguments
    ///
    /// * `lhs` - Hardware memory for left-hand side argument.
    /// * `rhs` - Hardware memory for right-hand side argument.
    /// * `dest` - Hardware memory for destination.
    /// * `num_elements` - Number of elements on each memory.
    ///
    /// # Safety
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
    /// * `num_elements` - Number of elements on each memory.
    ///
    /// # Safety
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
    /// * `num_elements` - Number of elements on each memory.
    ///
    /// # Safety
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
    /// * `num_elements` - Number of elements on each memory.
    ///
    /// # Safety
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

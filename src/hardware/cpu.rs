use std::alloc;

/// Default memory alignment for allocating buffers.
const DEFAULT_MEMORY_ALIGNMENT: usize = 8;

/// Hardware for computation on local CPUs.
///
/// Memories on this hardware are identical with the usual host memory and are allocated through
/// `GlobalAlloc`.
pub struct CpuHardware {
    /// Name of this hardware.
    name: String,
}

impl CpuHardware {
    /// Creates a new `CpuHardwareHardware` object.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of this hardware.
    ///
    /// # Returns
    ///
    /// A new `CpuHardwareHardware` object.
    pub(crate) fn new(name: &str) -> Self {
        Self {
            name: String::from(name),
        }
    }
}

unsafe impl crate::hardware::Hardware for CpuHardware {
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
        handle
    }

    unsafe fn deallocate_memory(&mut self, handle: *mut u8, size: usize) {
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

    unsafe fn fill_f32(&mut self, dest: *mut u8, value: f32, num_elements: usize) {
        let dest = dest as *mut f32;
        for i in 0..num_elements {
            *dest.add(i) = value;
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

#[cfg(test)]
mod tests {
    use crate::hardware::cpu::CpuHardware;
    use crate::hardware::Hardware;
    use std::mem::size_of;

    #[test]
    fn test_name() {
        assert_eq!(CpuHardware::new("test").name(), "test");
    }

    #[test]
    fn test_allocate_memory() {
        let mut hw = CpuHardware::new("test");
        unsafe {
            let ptr = hw.allocate_memory(4);
            let arr_ptr = &mut *(ptr as *mut [u8; 4]);

            // `ptr` should be a usual host memory.
            *arr_ptr = [0x01, 0x02, 0x03, 0x04];
            *ptr.add(0) += 0x40;
            *ptr.add(1) += 0x30;
            *ptr.add(2) += 0x20;
            *ptr.add(3) += 0x10;
            assert_eq!(*arr_ptr, [0x41, 0x32, 0x23, 0x14]);

            hw.deallocate_memory(ptr, 4);
        }
    }

    #[test]
    fn test_copy_host_to_hardware() {
        let mut hw = CpuHardware::new("test");
        unsafe {
            let dest = hw.allocate_memory(4);
            {
                // Constrains the `src`'s lifetime.
                let src: Vec<u8> = vec![1, 2, 3, 4];
                hw.copy_host_to_hardware(src.as_ptr(), dest, 4);
            }
            assert_eq!(*(dest as *const [u8; 4]), [1, 2, 3, 4]);
            hw.deallocate_memory(dest, 4);
        }
    }

    #[test]
    fn test_copy_hardware_to_host() {
        let mut hw = CpuHardware::new("test");
        unsafe {
            let mut dest: Vec<u8> = vec![0; 4];
            {
                // Constrains the `src`'s lifetime.
                let src = hw.allocate_memory(4);
                *(src as *mut [u8; 4]) = [1, 2, 3, 4];
                hw.copy_hardware_to_host(src, dest.as_mut_ptr(), 4);
                hw.deallocate_memory(src, 4);
            }
            assert_eq!(dest, vec![1, 2, 3, 4])
        }
    }

    #[test]
    fn test_fill_f32() {
        let mut hw = CpuHardware::new("test");
        unsafe {
            let dest = hw.allocate_memory(4 * size_of::<f32>());
            hw.fill_f32(dest, 42., 4);
            assert_eq!(*(dest as *const [f32; 4]), [42.; 4]);
            hw.deallocate_memory(dest, 4 * size_of::<f32>());
        }
    }

    #[test]
    fn test_elementwise_add_f32() {
        let mut hw = CpuHardware::new("test");
        let size = 4 * size_of::<f32>();
        unsafe {
            let lhs = hw.allocate_memory(size);
            let rhs = hw.allocate_memory(size);
            let dest = hw.allocate_memory(size);
            *(lhs as *mut [f32; 4]) = [1., 2., 3., 4.];
            *(rhs as *mut [f32; 4]) = [5., 6., 7., 8.];
            hw.elementwise_add_f32(lhs, rhs, dest, 4);
            assert_eq!(*(dest as *const [f32; 4]), [6., 8., 10., 12.]);
            hw.deallocate_memory(lhs, size);
            hw.deallocate_memory(rhs, size);
            hw.deallocate_memory(dest, size);
        }
    }

    #[test]
    fn test_elementwise_sub_f32() {
        let mut hw = CpuHardware::new("test");
        let size = 4 * size_of::<f32>();
        unsafe {
            let lhs = hw.allocate_memory(size);
            let rhs = hw.allocate_memory(size);
            let dest = hw.allocate_memory(size);
            *(lhs as *mut [f32; 4]) = [9., 8., 7., 6.];
            *(rhs as *mut [f32; 4]) = [1., 2., 3., 4.];
            hw.elementwise_sub_f32(lhs, rhs, dest, 4);
            assert_eq!(*(dest as *const [f32; 4]), [8., 6., 4., 2.]);
            hw.deallocate_memory(lhs, size);
            hw.deallocate_memory(rhs, size);
            hw.deallocate_memory(dest, size);
        }
    }

    #[test]
    fn test_elementwise_mul_f32() {
        let mut hw = CpuHardware::new("test");
        let size = 4 * size_of::<f32>();
        unsafe {
            let lhs = hw.allocate_memory(size);
            let rhs = hw.allocate_memory(size);
            let dest = hw.allocate_memory(size);
            *(lhs as *mut [f32; 4]) = [1., 2., 3., 4.];
            *(rhs as *mut [f32; 4]) = [5., 6., 7., 8.];
            hw.elementwise_mul_f32(lhs, rhs, dest, 4);
            assert_eq!(*(dest as *const [f32; 4]), [5., 12., 21., 32.]);
            hw.deallocate_memory(lhs, size);
            hw.deallocate_memory(rhs, size);
            hw.deallocate_memory(dest, size);
        }
    }

    #[test]
    fn test_elementwise_div_f32() {
        let mut hw = CpuHardware::new("test");
        let size = 4 * size_of::<f32>();
        unsafe {
            let lhs = hw.allocate_memory(size);
            let rhs = hw.allocate_memory(size);
            let dest = hw.allocate_memory(size);
            *(lhs as *mut [f32; 4]) = [1., 2., 3., 4.];
            *(rhs as *mut [f32; 4]) = [4., 2., 1., 0.5];
            hw.elementwise_div_f32(lhs, rhs, dest, 4);
            assert_eq!(*(dest as *const [f32; 4]), [0.25, 1., 3., 8.]);
            hw.deallocate_memory(lhs, size);
            hw.deallocate_memory(rhs, size);
            hw.deallocate_memory(dest, size);
        }
    }
}

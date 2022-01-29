use std::alloc;
use std::collections::HashSet;

use crate::hardware::Hardware;

/// Default memory alignment for allocating buffers.
const DEFAULT_MEMORY_ALIGNMENT: usize = 8;

/// Hardware for computation on local CPUs.
///
/// Memories on this hardware are identical with the usual host memory and are allocated through
/// `GlobalAlloc`.
pub struct CpuHardware {
    /// Name of this hardware.
    name: String,

    /// Registry of supplied pointer and associated memory size.
    supplied: HashSet<(usize, usize)>,
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
            supplied: HashSet::new(),
        }
    }
}

impl Drop for CpuHardware {
    fn drop(&mut self) {
        if self.supplied.len() > 0 {
            // Leak detected. Removes all pointers anyway.
            let num_leaked = self.supplied.len();

            while !self.supplied.is_empty() {
                unsafe {
                    let &(handle, size) = self.supplied.iter().next().unwrap();
                    self.deallocate_memory(handle as *mut u8, size);
                }
            }

            panic!(
                "Detected memory leak: {} memory blocks have not been released.",
                num_leaked,
            );
        }
    }
}

unsafe impl Hardware for CpuHardware {
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

        // Remembers only memory with nonzero length.
        if size > 0 {
            if !self.supplied.insert((handle as usize, size)) {
                // As we ignored zero-length memories, this condition should never be satisfied.
                panic!("Handle {:016p} is supplied twice.", handle);
            }
        }

        handle
    }

    unsafe fn deallocate_memory(&mut self, handle: *mut u8, size: usize) {
        // Removes only memory with nonzero length.
        if size > 0 {
            if !self.supplied.remove(&((handle as usize), size)) {
                panic!("Handle {:016p} was not supplied.", handle);
            }
        }

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
    use crate::buffer::Buffer;
    use crate::hardware::cpu::CpuHardware;
    use crate::hardware::Hardware;
    use std::mem::size_of;

    use std::sync::Mutex;

    /// Helper function to create mutex-guarded CpuHardwre.
    fn make_hardware() -> Mutex<Box<dyn Hardware>> {
        Mutex::new(Box::new(CpuHardware::new("test")))
    }

    #[test]
    fn test_name() {
        assert_eq!(CpuHardware::new("test").name(), "test");
    }

    #[test]
    fn test_allocate_memory() {
        let hw = make_hardware();

        unsafe {
            let mut buf = Buffer::raw(&hw, 4);

            // Owned memory in `buf` should be a usual host memory.
            *(buf.as_mut_handle() as *mut [u8; 4]) = [0x01, 0x02, 0x03, 0x04];
            *buf.as_mut_handle().add(0) += 0x40;
            *buf.as_mut_handle().add(1) += 0x30;
            *buf.as_mut_handle().add(2) += 0x20;
            *buf.as_mut_handle().add(3) += 0x10;
            assert_eq!(
                *(buf.as_handle() as *const [u8; 4]),
                [0x41, 0x32, 0x23, 0x14]
            );
        }
    }

    #[test]
    #[should_panic(expected = "Detected memory leak: 1 memory blocks have not been released.")]
    fn test_memory_leak() {
        let hw = make_hardware();
        unsafe {
            hw.lock().unwrap().allocate_memory(1);
        }
    }

    #[test]
    fn test_zero_memory_leak() {
        let hw = make_hardware();
        unsafe {
            let mut hw = hw.lock().unwrap();

            // The hardware don't care about zero-length memories.
            hw.allocate_memory(0);
            hw.allocate_memory(0);
            hw.allocate_memory(0);
        }
    }

    #[test]
    #[should_panic]
    fn test_deallocate_memory_twice() {
        let hw = make_hardware();
        unsafe {
            let mut hw = hw.lock().unwrap();
            let ptr = hw.allocate_memory(1);
            hw.deallocate_memory(ptr, 1);
            hw.deallocate_memory(ptr, 1);
        }
    }

    #[test]
    fn test_copy_host_to_hardware() {
        let hw = make_hardware();
        unsafe {
            let mut dest = Buffer::raw(&hw, 4);
            let src: Vec<u8> = vec![1, 2, 3, 4];
            hw.lock()
                .unwrap()
                .copy_host_to_hardware(src.as_ptr(), dest.as_mut_handle(), 4);
            assert_eq!(*(dest.as_handle() as *const [u8; 4]), [1, 2, 3, 4]);
        }
    }

    #[test]
    fn test_copy_hardware_to_host() {
        let hw = make_hardware();
        unsafe {
            let src = Buffer::raw(&hw, 4);
            let mut dest: Vec<u8> = vec![0; 4];
            *(src.as_handle() as *mut [u8; 4]) = [1, 2, 3, 4];
            hw.lock()
                .unwrap()
                .copy_hardware_to_host(src.as_handle(), dest.as_mut_ptr(), 4);
            assert_eq!(dest, vec![1, 2, 3, 4])
        }
    }

    #[test]
    fn test_fill_f32() {
        let hw = make_hardware();
        unsafe {
            let mut dest = Buffer::raw(&hw, 4 * size_of::<f32>());
            hw.lock().unwrap().fill_f32(dest.as_mut_handle(), 42., 4);
            assert_eq!(*(dest.as_handle() as *const [f32; 4]), [42.; 4]);
        }
    }

    #[test]
    fn test_elementwise_add_f32() {
        let hw = make_hardware();
        let size = 4 * size_of::<f32>();
        unsafe {
            let mut lhs = Buffer::raw(&hw, size);
            let mut rhs = Buffer::raw(&hw, size);
            let mut dest = Buffer::raw(&hw, size);
            *(lhs.as_mut_handle() as *mut [f32; 4]) = [1., 2., 3., 4.];
            *(rhs.as_mut_handle() as *mut [f32; 4]) = [5., 6., 7., 8.];
            hw.lock().unwrap().elementwise_add_f32(
                lhs.as_handle(),
                rhs.as_handle(),
                dest.as_mut_handle(),
                4,
            );
            assert_eq!(
                *(dest.as_mut_handle() as *const [f32; 4]),
                [6., 8., 10., 12.]
            );
        }
    }

    #[test]
    fn test_elementwise_sub_f32() {
        let hw = make_hardware();
        let size = 4 * size_of::<f32>();
        unsafe {
            let mut lhs = Buffer::raw(&hw, size);
            let mut rhs = Buffer::raw(&hw, size);
            let mut dest = Buffer::raw(&hw, size);
            *(lhs.as_mut_handle() as *mut [f32; 4]) = [9., 8., 7., 6.];
            *(rhs.as_mut_handle() as *mut [f32; 4]) = [1., 2., 3., 4.];
            hw.lock().unwrap().elementwise_sub_f32(
                lhs.as_handle(),
                rhs.as_handle(),
                dest.as_mut_handle(),
                4,
            );
            assert_eq!(*(dest.as_handle() as *const [f32; 4]), [8., 6., 4., 2.]);
        }
    }

    #[test]
    fn test_elementwise_mul_f32() {
        let hw = make_hardware();
        let size = 4 * size_of::<f32>();
        unsafe {
            let mut lhs = Buffer::raw(&hw, size);
            let mut rhs = Buffer::raw(&hw, size);
            let mut dest = Buffer::raw(&hw, size);
            *(lhs.as_mut_handle() as *mut [f32; 4]) = [1., 2., 3., 4.];
            *(rhs.as_mut_handle() as *mut [f32; 4]) = [5., 6., 7., 8.];
            hw.lock().unwrap().elementwise_mul_f32(
                lhs.as_handle(),
                rhs.as_handle(),
                dest.as_mut_handle(),
                4,
            );
            assert_eq!(*(dest.as_handle() as *const [f32; 4]), [5., 12., 21., 32.]);
        }
    }

    #[test]
    fn test_elementwise_div_f32() {
        let hw = make_hardware();
        let size = 4 * size_of::<f32>();
        unsafe {
            let mut lhs = Buffer::raw(&hw, size);
            let mut rhs = Buffer::raw(&hw, size);
            let mut dest = Buffer::raw(&hw, size);
            *(lhs.as_mut_handle() as *mut [f32; 4]) = [1., 2., 3., 4.];
            *(rhs.as_mut_handle() as *mut [f32; 4]) = [4., 2., 1., 0.5];
            hw.lock().unwrap().elementwise_div_f32(
                lhs.as_handle(),
                rhs.as_handle(),
                dest.as_mut_handle(),
                4,
            );
            assert_eq!(*(dest.as_handle() as *const [f32; 4]), [0.25, 1., 3., 8.]);
        }
    }
}

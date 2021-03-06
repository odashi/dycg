use crate::array::*;
use crate::hardware::cpu::CpuHardware;
use std::mem;
use std::ptr;

#[test]
fn test_raw_scalar() {
    let hw = RefCell::new(CpuHardware::new());
    let array = unsafe { Array::raw(&hw, Shape::new([])) };
    assert!(ptr::eq(array.hardware(), &hw));
    assert_eq!(array.buffer.size(), mem::size_of::<f32>());
    assert_eq!(array.shape, Shape::new([]));
}

#[test]
fn test_raw_0() {
    let hw = RefCell::new(CpuHardware::new());
    let array = unsafe { Array::raw(&hw, Shape::new([0])) };
    assert!(ptr::eq(array.hardware(), &hw));
    assert_eq!(array.buffer.size(), 0);
    assert_eq!(array.shape, Shape::new([0]));
}

#[test]
fn test_raw_n() {
    let hw = RefCell::new(CpuHardware::new());
    let array = unsafe { Array::raw(&hw, Shape::new([42])) };
    assert!(ptr::eq(array.hardware(), &hw));
    assert_eq!(array.buffer.size(), 42 * mem::size_of::<f32>());
    assert_eq!(array.shape, Shape::new([42]));
}

#[test]
fn test_raw_colocated_scalar() {
    let hw = RefCell::new(CpuHardware::new());
    let other = unsafe { Array::raw(&hw, Shape::new([])) };
    let colocated = unsafe { Array::raw_colocated(&other, Shape::new([])) };
    assert!(ptr::eq(colocated.hardware(), &hw));
    assert_eq!(colocated.buffer.size(), mem::size_of::<f32>());
    assert_eq!(colocated.shape, Shape::new([]));
}

#[test]
fn test_raw_colocated_0() {
    let hw = RefCell::new(CpuHardware::new());
    let other = unsafe { Array::raw(&hw, Shape::new([])) };
    let colocated = unsafe { Array::raw_colocated(&other, Shape::new([0])) };
    assert!(ptr::eq(colocated.hardware(), &hw));
    assert_eq!(colocated.buffer.size(), 0);
    assert_eq!(colocated.shape, Shape::new([0]));
}

#[test]
fn test_raw_colocated_n() {
    let hw = RefCell::new(CpuHardware::new());
    let other = unsafe { Array::raw(&hw, Shape::new([])) };
    let colocated = unsafe { Array::raw_colocated(&other, Shape::new([42])) };
    assert!(ptr::eq(colocated.hardware(), &hw));
    assert_eq!(colocated.buffer.size(), 42 * mem::size_of::<f32>());
    assert_eq!(colocated.shape, Shape::new([42]));
}

#[test]
fn test_shape() {
    let hw = RefCell::new(CpuHardware::new());
    let array = unsafe { Array::raw(&hw, Shape::new([])) };
    assert!(ptr::eq(array.shape(), &array.shape));
}

#[test]
fn test_set_scalar_f32() {
    let hw = RefCell::new(CpuHardware::new());
    let array = unsafe {
        let mut array = Array::raw(&hw, Shape::new([]));
        array.set_scalar_f32(123.);
        array
    };
    assert_eq!(array.get_scalar_f32(), Ok(123.));
    assert_eq!(array.get_values_f32(), vec![123.]);
}

#[test]
fn test_get_scalar_f32_1() {
    let hw = RefCell::new(CpuHardware::new());
    let array = unsafe { Array::raw(&hw, Shape::new([1])) };
    assert!(array.get_scalar_f32().is_err());
}

#[test]
fn test_get_scalar_f32_n() {
    let hw = RefCell::new(CpuHardware::new());
    let array = unsafe { Array::raw(&hw, Shape::new([42])) };
    assert!(array.get_scalar_f32().is_err());
}

#[test]
fn test_set_values_f32_scalar() {
    let hw = RefCell::new(CpuHardware::new());
    let mut array = unsafe { Array::raw(&hw, Shape::new([])) };
    array.set_values_f32(&[123.]).unwrap();
    assert_eq!(array.get_scalar_f32(), Ok(123.));
    assert_eq!(array.get_values_f32(), vec![123.]);

    assert!(array.set_values_f32(&[]).is_err());
    assert!(array.set_values_f32(&[123., 456.]).is_err());
    assert!(array.set_values_f32(&[123., 456., 789.]).is_err());
}

#[test]
fn test_set_values_f32_0() {
    let hw = RefCell::new(CpuHardware::new());
    let mut array = unsafe { Array::raw(&hw, Shape::new([0])) };
    array.set_values_f32(&[]).unwrap();
    assert_eq!(array.get_values_f32(), vec![]);

    assert!(array.set_values_f32(&[111.]).is_err());
    assert!(array.set_values_f32(&[111., 222.]).is_err());
}

#[test]
fn test_set_values_f32_n() {
    let hw = RefCell::new(CpuHardware::new());
    let mut array = unsafe { Array::raw(&hw, Shape::new([3])) };
    array.set_values_f32(&[123., 456., 789.]).unwrap();
    assert_eq!(array.get_values_f32(), vec![123., 456., 789.]);

    assert!(array.set_values_f32(&[]).is_err());
    assert!(array.set_values_f32(&[111.]).is_err());
    assert!(array.set_values_f32(&[111., 222.]).is_err());
    assert!(array.set_values_f32(&[111., 222., 333., 444.]).is_err());
}

#[test]
fn test_scalar_f32() {
    let hw = RefCell::new(CpuHardware::new());
    let array = 123f32.into_array(&hw);
    assert_eq!(array.shape, Shape::new([]));
    assert_eq!(array.get_scalar_f32(), Ok(123.));
    assert_eq!(array.get_values_f32(), vec![123.]);
}

#[test]
fn test_constant_f32_scalar() {
    let hw = RefCell::new(CpuHardware::new());
    let array = Array::constant_f32(&hw, Shape::new([]), &[123.]).unwrap();
    assert_eq!(array.shape, Shape::new([]));
    assert_eq!(array.get_scalar_f32(), Ok(123.));
    assert_eq!(array.get_values_f32(), vec![123.]);
}

#[test]
fn test_constant_f32_scalar_invalid() {
    let hw = RefCell::new(CpuHardware::new());
    assert!(Array::constant_f32(&hw, Shape::new([]), &[]).is_err());
    assert!(Array::constant_f32(&hw, Shape::new([]), &[111., 222.]).is_err());
}

#[test]
fn test_constant_f32_0() {
    let hw = RefCell::new(CpuHardware::new());
    let array = Array::constant_f32(&hw, Shape::new([0]), &[]).unwrap();
    assert_eq!(array.shape, Shape::new([0]));
    assert!(array.get_scalar_f32().is_err());
    assert_eq!(array.get_values_f32(), vec![]);
}

#[test]
fn test_constant_f32_0_invalid() {
    let hw = RefCell::new(CpuHardware::new());
    assert!(Array::constant_f32(&hw, Shape::new([0]), &[111.]).is_err());
    assert!(Array::constant_f32(&hw, Shape::new([0]), &[111., 222.]).is_err());
}

#[test]
fn test_constant_f32_n() {
    let hw = RefCell::new(CpuHardware::new());
    let array = Array::constant_f32(&hw, Shape::new([3]), &[123., 456., 789.]).unwrap();
    assert_eq!(array.shape, Shape::new([3]));
    assert!(array.get_scalar_f32().is_err());
    assert_eq!(array.get_values_f32(), vec![123., 456., 789.]);
}

#[test]
fn test_constant_f32_n_invalid() {
    let hw = RefCell::new(CpuHardware::new());
    assert!(Array::constant_f32(&hw, Shape::new([3]), &[]).is_err());
    assert!(Array::constant_f32(&hw, Shape::new([3]), &[111.]).is_err());
    assert!(Array::constant_f32(&hw, Shape::new([3]), &[111., 222.]).is_err());
    assert!(Array::constant_f32(&hw, Shape::new([3]), &[111., 222., 333., 444.]).is_err());
}

#[test]
fn test_fill_f32_scalar() {
    let hw = RefCell::new(CpuHardware::new());
    let array = Array::fill_f32(&hw, Shape::new([]), 123.);
    assert_eq!(array.shape, Shape::new([]));
    assert_eq!(array.get_scalar_f32(), Ok(123.));
    assert_eq!(array.get_values_f32(), vec![123.]);
}

#[test]
fn test_fill_f32_0() {
    let hw = RefCell::new(CpuHardware::new());
    let array = Array::fill_f32(&hw, Shape::new([0]), 123.);
    assert_eq!(array.shape, Shape::new([0]));
    assert!(array.get_scalar_f32().is_err());
    assert_eq!(array.get_values_f32(), vec![]);
}

#[test]
fn test_fill_f32_n() {
    let hw = RefCell::new(CpuHardware::new());
    let array = Array::fill_f32(&hw, Shape::new([3]), 123.);
    assert_eq!(array.shape, Shape::new([3]));
    assert!(array.get_scalar_f32().is_err());
    assert_eq!(array.get_values_f32(), vec![123., 123., 123.]);
}

#[test]
fn test_fill_colocated_f32_scalar() {
    let hw = RefCell::new(CpuHardware::new());
    let other = unsafe { Array::raw(&hw, Shape::new([])) };
    let array = Array::fill_colocated_f32(&other, Shape::new([]), 123.);
    assert_eq!(array.shape, Shape::new([]));
    assert!(ptr::eq(array.hardware(), &hw));
    assert_eq!(array.get_scalar_f32(), Ok(123.));
    assert_eq!(array.get_values_f32(), vec![123.]);
}

#[test]
fn test_fill_colocated_f32_0() {
    let hw = RefCell::new(CpuHardware::new());
    let other = unsafe { Array::raw(&hw, Shape::new([])) };
    let array = Array::fill_colocated_f32(&other, Shape::new([0]), 123.);
    assert_eq!(array.shape, Shape::new([0]));
    assert!(ptr::eq(array.hardware(), &hw));
    assert!(array.get_scalar_f32().is_err());
    assert_eq!(array.get_values_f32(), vec![]);
}

#[test]
fn test_fill_colocated_f32_n() {
    let hw = RefCell::new(CpuHardware::new());
    let other = unsafe { Array::raw(&hw, Shape::new([])) };
    let array = Array::fill_colocated_f32(&other, Shape::new([3]), 123.);
    assert_eq!(array.shape, Shape::new([3]));
    assert!(ptr::eq(array.hardware(), &hw));
    assert!(array.get_scalar_f32().is_err());
    assert_eq!(array.get_values_f32(), vec![123., 123., 123.]);
}

#[test]
fn test_elementwise_unary_f32_scalar() {
    let hw = RefCell::new(CpuHardware::new());
    let x = 123f32.into_array(&hw);

    let y = x.elementwise_neg_f32();
    assert_eq!(y.shape, Shape::new([]));
    assert!(ptr::eq(y.hardware(), &hw));
    assert_eq!(y.get_values_f32(), vec![-123.]);
}

#[test]
fn test_elementwise_unary_f32_0() {
    let hw = RefCell::new(CpuHardware::new());
    let x = Array::constant_f32(&hw, Shape::new([0]), &[]).unwrap();

    let y = x.elementwise_neg_f32();
    assert_eq!(y.shape, Shape::new([0]));
    assert!(ptr::eq(y.hardware(), &hw));
    assert_eq!(y.get_values_f32(), vec![]);
}

#[test]
fn test_elementwise_unary_f32_n() {
    let hw = RefCell::new(CpuHardware::new());
    let x = Array::constant_f32(&hw, Shape::new([3]), &[123., 456., 789.]).unwrap();

    let y = x.elementwise_neg_f32();
    assert_eq!(y.shape, Shape::new([3]));
    assert!(ptr::eq(y.hardware(), &hw));
    assert_eq!(y.get_values_f32(), vec![-123., -456., -789.]);
}

#[test]
fn test_elementwise_binary_f32_scalar_scalar() {
    let hw = RefCell::new(CpuHardware::new());
    let a = 111f32.into_array(&hw);
    let b = 222f32.into_array(&hw);

    let y = a.elementwise_add_f32(&b).unwrap();
    assert_eq!(y.shape, Shape::new([]));
    assert!(ptr::eq(y.hardware(), &hw));
    assert_eq!(y.get_values_f32(), vec![333.]);

    let y = a.elementwise_sub_f32(&b).unwrap();
    assert_eq!(y.shape, Shape::new([]));
    assert!(ptr::eq(y.hardware(), &hw));
    assert_eq!(y.get_values_f32(), vec![-111.]);

    let y = a.elementwise_mul_f32(&b).unwrap();
    assert_eq!(y.shape, Shape::new([]));
    assert!(ptr::eq(y.hardware(), &hw));
    assert_eq!(y.get_values_f32(), vec![24642.]);

    let y = a.elementwise_div_f32(&b).unwrap();
    assert_eq!(y.shape, Shape::new([]));
    assert!(ptr::eq(y.hardware(), &hw));
    assert_eq!(y.get_values_f32(), vec![0.5]);
}

#[test]
fn test_elementwise_binary_f32_scalar_self() {
    let hw = RefCell::new(CpuHardware::new());
    let a = 111f32.into_array(&hw);

    let y = a.elementwise_add_f32(&a).unwrap();
    assert_eq!(y.shape, Shape::new([]));
    assert!(ptr::eq(y.hardware(), &hw));
    assert_eq!(y.get_values_f32(), vec![222.]);

    let y = a.elementwise_sub_f32(&a).unwrap();
    assert_eq!(y.shape, Shape::new([]));
    assert!(ptr::eq(y.hardware(), &hw));
    assert_eq!(y.get_values_f32(), vec![0.]);

    let y = a.elementwise_mul_f32(&a).unwrap();
    assert_eq!(y.shape, Shape::new([]));
    assert!(ptr::eq(y.hardware(), &hw));
    assert_eq!(y.get_values_f32(), vec![12321.]);

    let y = a.elementwise_div_f32(&a).unwrap();
    assert_eq!(y.shape, Shape::new([]));
    assert!(ptr::eq(y.hardware(), &hw));
    assert_eq!(y.get_values_f32(), vec![1.]);
}

#[test]
fn test_elementwise_binary_f32_0_0() {
    let hw = RefCell::new(CpuHardware::new());
    let a = Array::constant_f32(&hw, Shape::new([0]), &[]).unwrap();
    let b = Array::constant_f32(&hw, Shape::new([0]), &[]).unwrap();

    let y = a.elementwise_add_f32(&b).unwrap();
    assert_eq!(y.shape, Shape::new([0]));
    assert!(ptr::eq(y.hardware(), &hw));
    assert_eq!(y.get_values_f32(), vec![]);

    let y = a.elementwise_sub_f32(&b).unwrap();
    assert_eq!(y.shape, Shape::new([0]));
    assert!(ptr::eq(y.hardware(), &hw));
    assert_eq!(y.get_values_f32(), vec![]);

    let y = a.elementwise_mul_f32(&b).unwrap();
    assert_eq!(y.shape, Shape::new([0]));
    assert!(ptr::eq(y.hardware(), &hw));
    assert_eq!(y.get_values_f32(), vec![]);

    let y = a.elementwise_div_f32(&b).unwrap();
    assert_eq!(y.shape, Shape::new([0]));
    assert!(ptr::eq(y.hardware(), &hw));
    assert_eq!(y.get_values_f32(), vec![]);
}

#[test]
fn test_elementwise_binary_f32_0_self() {
    let hw = RefCell::new(CpuHardware::new());
    let a = Array::constant_f32(&hw, Shape::new([0]), &[]).unwrap();

    let y = a.elementwise_add_f32(&a).unwrap();
    assert_eq!(y.shape, Shape::new([0]));
    assert!(ptr::eq(y.hardware(), &hw));
    assert_eq!(y.get_values_f32(), vec![]);

    let y = a.elementwise_sub_f32(&a).unwrap();
    assert_eq!(y.shape, Shape::new([0]));
    assert!(ptr::eq(y.hardware(), &hw));
    assert_eq!(y.get_values_f32(), vec![]);

    let y = a.elementwise_mul_f32(&a).unwrap();
    assert_eq!(y.shape, Shape::new([0]));
    assert!(ptr::eq(y.hardware(), &hw));
    assert_eq!(y.get_values_f32(), vec![]);

    let y = a.elementwise_div_f32(&a).unwrap();
    assert_eq!(y.shape, Shape::new([0]));
    assert!(ptr::eq(y.hardware(), &hw));
    assert_eq!(y.get_values_f32(), vec![]);
}

#[test]
fn test_elementwise_binary_f32_n_n() {
    let hw = RefCell::new(CpuHardware::new());
    let a = Array::constant_f32(&hw, Shape::new([3]), &[111., 222., 333.]).unwrap();
    let b = Array::constant_f32(&hw, Shape::new([3]), &[444., 555., 666.]).unwrap();

    let y = a.elementwise_add_f32(&b).unwrap();
    assert_eq!(y.shape, Shape::new([3]));
    assert!(ptr::eq(y.hardware(), &hw));
    assert_eq!(y.get_values_f32(), vec![555., 777., 999.]);

    let y = a.elementwise_sub_f32(&b).unwrap();
    assert_eq!(y.shape, Shape::new([3]));
    assert!(ptr::eq(y.hardware(), &hw));
    assert_eq!(y.get_values_f32(), vec![-333., -333., -333.]);

    let y = a.elementwise_mul_f32(&b).unwrap();
    assert_eq!(y.shape, Shape::new([3]));
    assert!(ptr::eq(y.hardware(), &hw));
    assert_eq!(y.get_values_f32(), vec![49284., 123210., 221778.]);

    let y = a.elementwise_div_f32(&b).unwrap();
    assert_eq!(y.shape, Shape::new([3]));
    assert!(ptr::eq(y.hardware(), &hw));
    // Attempting exact matching even for 0.4.
    assert_eq!(y.get_values_f32(), vec![0.25, 0.4, 0.5]);
}

#[test]
fn test_elementwise_binary_f32_n_self() {
    let hw = RefCell::new(CpuHardware::new());
    let a = Array::constant_f32(&hw, Shape::new([3]), &[111., 222., 333.]).unwrap();

    let y = a.elementwise_add_f32(&a).unwrap();
    assert_eq!(y.shape, Shape::new([3]));
    assert!(ptr::eq(y.hardware(), &hw));
    assert_eq!(y.get_values_f32(), vec![222., 444., 666.]);

    let y = a.elementwise_sub_f32(&a).unwrap();
    assert_eq!(y.shape, Shape::new([3]));
    assert!(ptr::eq(y.hardware(), &hw));
    assert_eq!(y.get_values_f32(), vec![0., 0., 0.]);

    let y = a.elementwise_mul_f32(&a).unwrap();
    assert_eq!(y.shape, Shape::new([3]));
    assert!(ptr::eq(y.hardware(), &hw));
    assert_eq!(y.get_values_f32(), vec![12321., 49284., 110889.]);

    let y = a.elementwise_div_f32(&a).unwrap();
    assert_eq!(y.shape, Shape::new([3]));
    assert!(ptr::eq(y.hardware(), &hw));
    assert_eq!(y.get_values_f32(), vec![1., 1., 1.]);
}

#[test]
fn test_elementwise_binary_f32_scalar_0() {
    let hw = RefCell::new(CpuHardware::new());
    let a = 111f32.into_array(&hw);
    let b = Array::constant_f32(&hw, Shape::new([0]), &[]).unwrap();

    assert!(a.elementwise_add_f32(&b).is_err());
    assert!(a.elementwise_sub_f32(&b).is_err());
    assert!(a.elementwise_mul_f32(&b).is_err());
    assert!(a.elementwise_div_f32(&b).is_err());

    assert!(b.elementwise_add_f32(&a).is_err());
    assert!(b.elementwise_sub_f32(&a).is_err());
    assert!(b.elementwise_mul_f32(&a).is_err());
    assert!(b.elementwise_div_f32(&a).is_err());
}

#[test]
fn test_elementwise_binary_f32_scalar_1() {
    let hw = RefCell::new(CpuHardware::new());
    let a = 111f32.into_array(&hw);
    let b = Array::constant_f32(&hw, Shape::new([1]), &[444.]).unwrap();

    assert!(a.elementwise_add_f32(&b).is_err());
    assert!(a.elementwise_sub_f32(&b).is_err());
    assert!(a.elementwise_mul_f32(&b).is_err());
    assert!(a.elementwise_div_f32(&b).is_err());

    assert!(b.elementwise_add_f32(&a).is_err());
    assert!(b.elementwise_sub_f32(&a).is_err());
    assert!(b.elementwise_mul_f32(&a).is_err());
    assert!(b.elementwise_div_f32(&a).is_err());
}

#[test]
fn test_elementwise_binary_f32_scalar_n() {
    let hw = RefCell::new(CpuHardware::new());
    let a = 111f32.into_array(&hw);
    let b = Array::constant_f32(&hw, Shape::new([3]), &[444., 555., 666.]).unwrap();

    assert!(a.elementwise_add_f32(&b).is_err());
    assert!(a.elementwise_sub_f32(&b).is_err());
    assert!(a.elementwise_mul_f32(&b).is_err());
    assert!(a.elementwise_div_f32(&b).is_err());

    assert!(b.elementwise_add_f32(&a).is_err());
    assert!(b.elementwise_sub_f32(&a).is_err());
    assert!(b.elementwise_mul_f32(&a).is_err());
    assert!(b.elementwise_div_f32(&a).is_err());
}

#[test]
fn test_elementwise_binary_f32_0_1() {
    let hw = RefCell::new(CpuHardware::new());
    let a = Array::constant_f32(&hw, Shape::new([0]), &[]).unwrap();
    let b = Array::constant_f32(&hw, Shape::new([1]), &[111.]).unwrap();

    assert!(a.elementwise_add_f32(&b).is_err());
    assert!(a.elementwise_sub_f32(&b).is_err());
    assert!(a.elementwise_mul_f32(&b).is_err());
    assert!(a.elementwise_div_f32(&b).is_err());

    assert!(b.elementwise_add_f32(&a).is_err());
    assert!(b.elementwise_sub_f32(&a).is_err());
    assert!(b.elementwise_mul_f32(&a).is_err());
    assert!(b.elementwise_div_f32(&a).is_err());
}

#[test]
fn test_elementwise_binary_f32_0_n() {
    let hw = RefCell::new(CpuHardware::new());
    let a = Array::constant_f32(&hw, Shape::new([0]), &[]).unwrap();
    let b = Array::constant_f32(&hw, Shape::new([3]), &[111., 222., 333.]).unwrap();

    assert!(a.elementwise_add_f32(&b).is_err());
    assert!(a.elementwise_sub_f32(&b).is_err());
    assert!(a.elementwise_mul_f32(&b).is_err());
    assert!(a.elementwise_div_f32(&b).is_err());

    assert!(b.elementwise_add_f32(&a).is_err());
    assert!(b.elementwise_sub_f32(&a).is_err());
    assert!(b.elementwise_mul_f32(&a).is_err());
    assert!(b.elementwise_div_f32(&a).is_err());
}

#[test]
fn test_elementwise_binary_f32_1_n() {
    let hw = RefCell::new(CpuHardware::new());
    let a = Array::constant_f32(&hw, Shape::new([1]), &[111.]).unwrap();
    let b = Array::constant_f32(&hw, Shape::new([3]), &[444., 555., 666.]).unwrap();

    assert!(a.elementwise_add_f32(&b).is_err());
    assert!(a.elementwise_sub_f32(&b).is_err());
    assert!(a.elementwise_mul_f32(&b).is_err());
    assert!(a.elementwise_div_f32(&b).is_err());

    assert!(b.elementwise_add_f32(&a).is_err());
    assert!(b.elementwise_sub_f32(&a).is_err());
    assert!(b.elementwise_mul_f32(&a).is_err());
    assert!(b.elementwise_div_f32(&a).is_err());
}

#[test]
fn test_elementwise_binary_f32_colocation() {
    let hw1 = RefCell::new(CpuHardware::new());
    let hw2 = RefCell::new(CpuHardware::new());
    let a = 123f32.into_array(&hw1);
    let b = 123f32.into_array(&hw2);

    assert!(a.elementwise_add_f32(&b).is_err());
    assert!(a.elementwise_sub_f32(&b).is_err());
    assert!(a.elementwise_mul_f32(&b).is_err());
    assert!(a.elementwise_div_f32(&b).is_err());

    assert!(b.elementwise_add_f32(&a).is_err());
    assert!(b.elementwise_sub_f32(&a).is_err());
    assert!(b.elementwise_mul_f32(&a).is_err());
    assert!(b.elementwise_div_f32(&a).is_err());
}

#[test]
fn test_clone_scalar() {
    let hw = RefCell::new(CpuHardware::new());
    let x = 123f32.into_array(&hw);
    let y = x.clone();
    assert_eq!(y.shape, Shape::new([]));
    assert!(ptr::eq(y.hardware(), &hw));
    assert_eq!(y.get_values_f32(), vec![123.]);
}

#[test]
fn test_clone_0() {
    let hw = RefCell::new(CpuHardware::new());
    let x = Array::constant_f32(&hw, Shape::new([0]), &[]).unwrap();
    let y = x.clone();
    assert_eq!(y.shape, Shape::new([0]));
    assert!(ptr::eq(y.hardware(), &hw));
    assert_eq!(y.get_values_f32(), vec![]);
}

#[test]
fn test_clone_n() {
    let hw = RefCell::new(CpuHardware::new());
    let x = Array::constant_f32(&hw, Shape::new([3]), &[123., 456., 789.]).unwrap();
    let y = x.clone();
    assert_eq!(y.shape, Shape::new([3]));
    assert!(ptr::eq(y.hardware(), &hw));
    assert_eq!(y.get_values_f32(), vec![123., 456., 789.]);
}

use crate::array::*;
use crate::hardware::cpu::CpuHardware;

#[test]
fn test_try_into_array0() {
    let hw = RefCell::new(CpuHardware::new());
    let src = Array::scalar_f32(&hw, 42.);
    let dest = ndarray::Array0::<f32>::try_from(&src).unwrap();
    assert_eq!(dest, ndarray::arr0(42.));
}

#[test]
fn test_try_into_array0_fail() {
    let hw = RefCell::new(CpuHardware::new());
    let src = Array::fill_f32(&hw, Shape::new([1]), 42.);
    assert!(ndarray::Array0::<f32>::try_from(&src).is_err());
}

#[test]
fn test_try_into_array1_0() {
    let hw = RefCell::new(CpuHardware::new());
    let src = Array::fill_f32(&hw, Shape::new([0]), 42.);
    let dest = ndarray::Array1::<f32>::try_from(&src).unwrap();
    assert_eq!(dest, ndarray::arr1(&[]));
}

#[test]
fn test_try_into_array1_n() {
    let hw = RefCell::new(CpuHardware::new());
    let src = Array::fill_f32(&hw, Shape::new([3]), 42.);
    let dest = ndarray::Array1::<f32>::try_from(&src).unwrap();
    assert_eq!(dest, ndarray::arr1(&[42., 42., 42.]));
}

#[test]
fn test_try_into_array1_fail() {
    let hw = RefCell::new(CpuHardware::new());
    let src1 = Array::fill_f32(&hw, Shape::new([]), 42.);
    let src2 = Array::fill_f32(&hw, Shape::new([1, 1]), 42.);
    assert!(ndarray::Array1::<f32>::try_from(&src1).is_err());
    assert!(ndarray::Array1::<f32>::try_from(&src2).is_err());
}

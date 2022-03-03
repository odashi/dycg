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

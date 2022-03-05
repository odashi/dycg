use crate::array::*;
use crate::hardware::cpu::CpuHardware;

#[test]
fn test_try_into_ndarray0() {
    let hw = RefCell::new(CpuHardware::new());
    let src = 42f32.into_array(&hw);
    let dest = ndarray::Array0::<f32>::try_from(&src).unwrap();
    assert_eq!(dest, ndarray::arr0(42.));
}

#[test]
fn test_try_into_ndarray0_fail() {
    let hw = RefCell::new(CpuHardware::new());
    for n in [1, 2, 3, 4, 5, 6, 7, 8] {
        let src = Array::fill_f32(&hw, Shape::from_slice(&vec![1; n]), 42.);
        assert!(ndarray::Array0::<f32>::try_from(&src).is_err());
    }
}

#[test]
fn test_try_into_ndarray1_0() {
    let hw = RefCell::new(CpuHardware::new());
    let src = Array::fill_f32(&hw, Shape::new([0]), 42.);
    let dest = ndarray::Array1::<f32>::try_from(&src).unwrap();
    assert_eq!(dest, ndarray::arr1(&[]));
}

#[test]
fn test_try_into_ndarray1_3() {
    let hw = RefCell::new(CpuHardware::new());
    let src = Array::fill_f32(&hw, Shape::new([3]), 42.);
    let dest = ndarray::Array1::<f32>::try_from(&src).unwrap();
    assert_eq!(dest, ndarray::arr1(&[42.; 3]));
}

#[test]
fn test_try_into_ndarray1_fail() {
    let hw = RefCell::new(CpuHardware::new());
    for n in [0, 2, 3, 4, 5, 6, 7, 8] {
        let src = Array::fill_f32(&hw, Shape::from_slice(&vec![1; n]), 42.);
        assert!(ndarray::Array1::<f32>::try_from(&src).is_err());
    }
}

#[test]
fn test_try_into_ndarray2_0x0() {
    let hw = RefCell::new(CpuHardware::new());
    let src = Array::fill_f32(&hw, Shape::new([0, 0]), 42.);
    let dest = ndarray::Array2::<f32>::try_from(&src).unwrap();
    assert_eq!(dest, ndarray::arr2(&[[42.; 0]; 0]));
}

#[test]
fn test_try_into_ndarray2_2x3() {
    let hw = RefCell::new(CpuHardware::new());
    let src = Array::fill_f32(&hw, Shape::new([2, 3]), 42.);
    let dest = ndarray::Array2::<f32>::try_from(&src).unwrap();
    assert_eq!(dest, ndarray::arr2(&[[42.; 3]; 2]));
}

#[test]
fn test_try_into_ndarray2_fail() {
    let hw = RefCell::new(CpuHardware::new());
    for n in [0, 1, 3, 4, 5, 6, 7, 8] {
        let src = Array::fill_f32(&hw, Shape::from_slice(&vec![1; n]), 42.);
        assert!(ndarray::Array2::<f32>::try_from(&src).is_err());
    }
}

#[test]
fn test_try_into_ndarray3_0x0x0() {
    let hw = RefCell::new(CpuHardware::new());
    let src = Array::fill_f32(&hw, Shape::new([0, 0, 0]), 42.);
    let dest = ndarray::Array3::<f32>::try_from(&src).unwrap();
    assert_eq!(dest, ndarray::arr3(&[[[42.; 0]; 0]; 0]));
}

#[test]
fn test_try_into_ndarray3_2x3x4() {
    let hw = RefCell::new(CpuHardware::new());
    let src = Array::fill_f32(&hw, Shape::new([2, 3, 4]), 42.);
    let dest = ndarray::Array3::<f32>::try_from(&src).unwrap();
    assert_eq!(dest, ndarray::arr3(&[[[42.; 4]; 3]; 2]));
}

#[test]
fn test_try_into_ndarray3_fail() {
    let hw = RefCell::new(CpuHardware::new());
    for n in [0, 1, 2, 4, 5, 6, 7, 8] {
        let src = Array::fill_f32(&hw, Shape::from_slice(&vec![1; n]), 42.);
        assert!(ndarray::Array3::<f32>::try_from(&src).is_err());
    }
}

#[test]
fn test_try_into_ndarray4_0x0x0x0() {
    let hw = RefCell::new(CpuHardware::new());
    let src = Array::fill_f32(&hw, Shape::new([0, 0, 0, 0]), 42.);
    let dest = ndarray::Array4::<f32>::try_from(&src).unwrap();
    assert_eq!(
        dest,
        ndarray::Array4::<f32>::from_shape_vec((0, 0, 0, 0), vec![]).unwrap()
    );
}

#[test]
fn test_try_into_ndarray4_2x3x4x5() {
    let hw = RefCell::new(CpuHardware::new());
    let src = Array::fill_f32(&hw, Shape::new([2, 3, 4, 5]), 42.);
    let dest = ndarray::Array4::<f32>::try_from(&src).unwrap();
    assert_eq!(
        dest,
        ndarray::Array4::<f32>::from_shape_vec((2, 3, 4, 5), vec![42.; 2 * 3 * 4 * 5]).unwrap()
    );
}

#[test]
fn test_try_into_ndarray4_fail() {
    let hw = RefCell::new(CpuHardware::new());
    for n in [0, 1, 2, 3, 5, 6, 7, 8] {
        let src = Array::fill_f32(&hw, Shape::from_slice(&vec![1; n]), 42.);
        assert!(ndarray::Array4::<f32>::try_from(&src).is_err());
    }
}

#[test]
fn test_try_into_ndarray5_0x0x0x0x0() {
    let hw = RefCell::new(CpuHardware::new());
    let src = Array::fill_f32(&hw, Shape::new([0, 0, 0, 0, 0]), 42.);
    let dest = ndarray::Array5::<f32>::try_from(&src).unwrap();
    assert_eq!(
        dest,
        ndarray::Array5::<f32>::from_shape_vec((0, 0, 0, 0, 0), vec![]).unwrap()
    );
}

#[test]
fn test_try_into_ndarray5_2x3x4x5x6() {
    let hw = RefCell::new(CpuHardware::new());
    let src = Array::fill_f32(&hw, Shape::new([2, 3, 4, 5, 6]), 42.);
    let dest = ndarray::Array5::<f32>::try_from(&src).unwrap();
    assert_eq!(
        dest,
        ndarray::Array5::<f32>::from_shape_vec((2, 3, 4, 5, 6), vec![42.; 2 * 3 * 4 * 5 * 6])
            .unwrap()
    );
}

#[test]
fn test_try_into_ndarray5_fail() {
    let hw = RefCell::new(CpuHardware::new());
    for n in [0, 1, 2, 3, 4, 6, 7, 8] {
        let src = Array::fill_f32(&hw, Shape::from_slice(&vec![1; n]), 42.);
        assert!(ndarray::Array5::<f32>::try_from(&src).is_err());
    }
}

#[test]
fn test_try_into_ndarray6_0x0x0x0x0x0() {
    let hw = RefCell::new(CpuHardware::new());
    let src = Array::fill_f32(&hw, Shape::new([0, 0, 0, 0, 0, 0]), 42.);
    let dest = ndarray::Array6::<f32>::try_from(&src).unwrap();
    assert_eq!(
        dest,
        ndarray::Array6::<f32>::from_shape_vec((0, 0, 0, 0, 0, 0), vec![]).unwrap()
    );
}

#[test]
fn test_try_into_ndarray6_2x3x4x5x6x7() {
    let hw = RefCell::new(CpuHardware::new());
    let src = Array::fill_f32(&hw, Shape::new([2, 3, 4, 5, 6, 7]), 42.);
    let dest = ndarray::Array6::<f32>::try_from(&src).unwrap();
    assert_eq!(
        dest,
        ndarray::Array6::<f32>::from_shape_vec(
            (2, 3, 4, 5, 6, 7),
            vec![42.; 2 * 3 * 4 * 5 * 6 * 7]
        )
        .unwrap()
    );
}

#[test]
fn test_try_into_ndarray6_fail() {
    let hw = RefCell::new(CpuHardware::new());
    for n in [0, 1, 2, 3, 4, 5, 7, 8] {
        let src = Array::fill_f32(&hw, Shape::from_slice(&vec![1; n]), 42.);
        assert!(ndarray::Array6::<f32>::try_from(&src).is_err());
    }
}

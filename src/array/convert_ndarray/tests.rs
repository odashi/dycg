use crate::array::*;
use crate::hardware::cpu::CpuHardware;
use ndarray::ShapeBuilder;

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

// Helper function to generate float range vector [0, end) with step size of 1.
fn iota(end: i32) -> Vec<f32> {
    (0..end).map(|x| x as f32).collect()
}

#[test]
fn test_from_ndarray0() {
    let hw = RefCell::new(CpuHardware::new());
    let src = ndarray::arr0(123.);
    let dest = src.into_array(&hw);
    assert_eq!(*dest.shape(), Shape::new([]));
    assert_eq!(dest.get_values_f32(), vec![123.]);
}

#[test]
fn test_from_ndarray1() {
    let hw = RefCell::new(CpuHardware::new());
    let src = ndarray::arr1(&iota(3));
    let dest = src.into_array(&hw);
    assert_eq!(*dest.shape(), Shape::new([3]));
    assert_eq!(dest.get_values_f32(), iota(3));
}

#[test]
fn test_from_ndarray1_0() {
    let hw = RefCell::new(CpuHardware::new());
    let src = ndarray::arr1(&[]);
    let dest = src.into_array(&hw);
    assert_eq!(*dest.shape(), Shape::new([0]));
    assert_eq!(dest.get_values_f32(), vec![]);
}

#[test]
fn test_from_ndarray2() {
    let hw = RefCell::new(CpuHardware::new());
    let src = ndarray::Array2::<f32>::from_shape_vec((2, 2), iota(4)).unwrap();
    let dest = src.into_array(&hw);
    assert_eq!(*dest.shape(), Shape::new([2, 2]));
    assert_eq!(dest.get_values_f32(), iota(4));
}

#[test]
fn test_from_ndarray2_0() {
    let hw = RefCell::new(CpuHardware::new());
    let src = ndarray::Array2::<f32>::from_shape_vec((0, 0), vec![]).unwrap();
    let dest = src.into_array(&hw);
    assert_eq!(*dest.shape(), Shape::new([0, 0]));
    assert_eq!(dest.get_values_f32(), vec![]);
}

#[test]
fn test_from_ndarray2_f() {
    let hw = RefCell::new(CpuHardware::new());
    let src = ndarray::Array2::<f32>::from_shape_vec((2, 2).f(), iota(4)).unwrap();
    let dest = src.into_array(&hw);
    assert_eq!(*dest.shape(), Shape::new([2, 2]));
    assert_eq!(dest.get_values_f32(), vec![0., 2., 1., 3.]);
}

#[test]
fn test_from_ndarray3() {
    let hw = RefCell::new(CpuHardware::new());
    let src = ndarray::Array3::<f32>::from_shape_vec((2, 2, 2), iota(8)).unwrap();
    let dest = src.into_array(&hw);
    assert_eq!(*dest.shape(), Shape::new([2, 2, 2]));
    assert_eq!(dest.get_values_f32(), iota(8));
}

#[test]
fn test_from_ndarray3_0() {
    let hw = RefCell::new(CpuHardware::new());
    let src = ndarray::Array3::<f32>::from_shape_vec((0, 0, 0), vec![]).unwrap();
    let dest = src.into_array(&hw);
    assert_eq!(*dest.shape(), Shape::new([0, 0, 0]));
    assert_eq!(dest.get_values_f32(), vec![]);
}

#[test]
fn test_from_ndarray3_f() {
    let hw = RefCell::new(CpuHardware::new());
    let src = ndarray::Array3::<f32>::from_shape_vec((2, 2, 2).f(), iota(8)).unwrap();
    let dest = src.into_array(&hw);
    assert_eq!(*dest.shape(), Shape::new([2, 2, 2]));
    assert_eq!(dest.get_values_f32(), vec![0., 4., 2., 6., 1., 5., 3., 7.]);
}

#[test]
fn test_from_ndarray4() {
    let hw = RefCell::new(CpuHardware::new());
    let src = ndarray::Array4::<f32>::from_shape_vec((2, 2, 2, 2), iota(16)).unwrap();
    let dest = src.into_array(&hw);
    assert_eq!(*dest.shape(), Shape::new([2, 2, 2, 2]));
    assert_eq!(dest.get_values_f32(), iota(16));
}

#[test]
fn test_from_ndarray4_0() {
    let hw = RefCell::new(CpuHardware::new());
    let src = ndarray::Array4::<f32>::from_shape_vec((0, 0, 0, 0), vec![]).unwrap();
    let dest = src.into_array(&hw);
    assert_eq!(*dest.shape(), Shape::new([0, 0, 0, 0]));
    assert_eq!(dest.get_values_f32(), vec![]);
}

#[test]
fn test_from_ndarray4_f() {
    let hw = RefCell::new(CpuHardware::new());
    let src = ndarray::Array4::<f32>::from_shape_vec((2, 2, 2, 2).f(), iota(16)).unwrap();
    let dest = src.into_array(&hw);
    assert_eq!(*dest.shape(), Shape::new([2, 2, 2, 2]));
    #[rustfmt::skip]
    assert_eq!(
        dest.get_values_f32(),
        vec![
            0., 8., 4., 12., 2., 10., 6., 14.,
            1., 9., 5., 13., 3., 11., 7., 15.,
        ]
    );
}

#[test]
fn test_from_ndarray5() {
    let hw = RefCell::new(CpuHardware::new());
    let src = ndarray::Array5::<f32>::from_shape_vec((2, 2, 2, 2, 2), iota(32)).unwrap();
    let dest = src.into_array(&hw);
    assert_eq!(*dest.shape(), Shape::new([2, 2, 2, 2, 2]));
    assert_eq!(dest.get_values_f32(), iota(32));
}

#[test]
fn test_from_ndarray5_0() {
    let hw = RefCell::new(CpuHardware::new());
    let src = ndarray::Array5::<f32>::from_shape_vec((0, 0, 0, 0, 0), vec![]).unwrap();
    let dest = src.into_array(&hw);
    assert_eq!(*dest.shape(), Shape::new([0, 0, 0, 0, 0]));
    assert_eq!(dest.get_values_f32(), vec![]);
}

#[test]
fn test_from_ndarray5_f() {
    let hw = RefCell::new(CpuHardware::new());
    let src = ndarray::Array5::<f32>::from_shape_vec((2, 2, 2, 2, 2).f(), iota(32)).unwrap();
    let dest = src.into_array(&hw);
    assert_eq!(*dest.shape(), Shape::new([2, 2, 2, 2, 2]));
    #[rustfmt::skip]
    assert_eq!(
        dest.get_values_f32(),
        vec![
            0., 16.,  8., 24., 4., 20., 12., 28.,
            2., 18., 10., 26., 6., 22., 14., 30.,
            1., 17.,  9., 25., 5., 21., 13., 29.,
            3., 19., 11., 27., 7., 23., 15., 31.,
        ]
    );
}

#[test]
fn test_from_ndarray6() {
    let hw = RefCell::new(CpuHardware::new());
    let src = ndarray::Array6::<f32>::from_shape_vec((2, 2, 2, 2, 2, 2), iota(64)).unwrap();
    let dest = src.into_array(&hw);
    assert_eq!(*dest.shape(), Shape::new([2, 2, 2, 2, 2, 2]));
    assert_eq!(dest.get_values_f32(), iota(64));
}

#[test]
fn test_from_ndarray6_0() {
    let hw = RefCell::new(CpuHardware::new());
    let src = ndarray::Array6::<f32>::from_shape_vec((0, 0, 0, 0, 0, 0), vec![]).unwrap();
    let dest = src.into_array(&hw);
    assert_eq!(*dest.shape(), Shape::new([0, 0, 0, 0, 0, 0]));
    assert_eq!(dest.get_values_f32(), vec![]);
}

#[test]
fn test_from_ndarray6_f() {
    let hw = RefCell::new(CpuHardware::new());
    let src = ndarray::Array6::<f32>::from_shape_vec((2, 2, 2, 2, 2, 2).f(), iota(64)).unwrap();
    let dest = src.into_array(&hw);
    assert_eq!(*dest.shape(), Shape::new([2, 2, 2, 2, 2, 2]));
    #[rustfmt::skip]
    assert_eq!(
        dest.get_values_f32(),
        vec![
            0., 32., 16., 48.,  8., 40., 24., 56.,
            4., 36., 20., 52., 12., 44., 28., 60.,
            2., 34., 18., 50., 10., 42., 26., 58.,
            6., 38., 22., 54., 14., 46., 30., 62.,
            1., 33., 17., 49.,  9., 41., 25., 57.,
            5., 37., 21., 53., 13., 45., 29., 61.,
            3., 35., 19., 51., 11., 43., 27., 59.,
            7., 39., 23., 55., 15., 47., 31., 63.,
        ]
    );
}

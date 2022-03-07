use crate::hardware::cpu::CpuHardware;
use crate::node::*;

#[test]
fn test_try_into_array0() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());
    let src = 42f32.into_node(&g, &hw);
    let dest = ndarray::Array0::<f32>::try_from(src).unwrap();
    assert_eq!(dest, ndarray::arr0(42.));
}

#[test]
fn test_try_into_array0_fail() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());
    for n in [1, 2, 3, 4, 5, 6, 7, 8] {
        let src = Node::fill(&g, &hw, Shape::from_slice(&vec![1; n]), 42.);
        assert!(ndarray::Array0::<f32>::try_from(src).is_err());
    }
}

#[test]
fn test_try_into_array1_0() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());
    let src = Node::fill(&g, &hw, Shape::new([0]), 42.);
    let dest = ndarray::Array1::<f32>::try_from(src).unwrap();
    assert_eq!(dest, ndarray::arr1(&[]));
}

#[test]
fn test_try_into_array1_3() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());
    let src = Node::fill(&g, &hw, Shape::new([3]), 42.);
    let dest = ndarray::Array1::<f32>::try_from(src).unwrap();
    assert_eq!(dest, ndarray::arr1(&[42.; 3]));
}

#[test]
fn test_try_into_array1_fail() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());
    for n in [0, 2, 3, 4, 5, 6, 7, 8] {
        let src = Node::fill(&g, &hw, Shape::from_slice(&vec![1; n]), 42.);
        assert!(ndarray::Array1::<f32>::try_from(src).is_err());
    }
}

#[test]
fn test_try_into_array2_0x0() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());
    let src = Node::fill(&g, &hw, Shape::new([0, 0]), 42.);
    let dest = ndarray::Array2::<f32>::try_from(src).unwrap();
    assert_eq!(dest, ndarray::arr2(&[[42.; 0]; 0]));
}

#[test]
fn test_try_into_array2_2x3() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());
    let src = Node::fill(&g, &hw, Shape::new([2, 3]), 42.);
    let dest = ndarray::Array2::<f32>::try_from(src).unwrap();
    assert_eq!(dest, ndarray::arr2(&[[42.; 3]; 2]));
}

#[test]
fn test_try_into_array2_fail() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());
    for n in [0, 1, 3, 4, 5, 6, 7, 8] {
        let src = Node::fill(&g, &hw, Shape::from_slice(&vec![1; n]), 42.);
        assert!(ndarray::Array2::<f32>::try_from(src).is_err());
    }
}

#[test]
fn test_try_into_array3_0x0x0() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());
    let src = Node::fill(&g, &hw, Shape::new([0, 0, 0]), 42.);
    let dest = ndarray::Array3::<f32>::try_from(src).unwrap();
    assert_eq!(dest, ndarray::arr3(&[[[42.; 0]; 0]; 0]));
}

#[test]
fn test_try_into_array3_2x3x4() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());
    let src = Node::fill(&g, &hw, Shape::new([2, 3, 4]), 42.);
    let dest = ndarray::Array3::<f32>::try_from(src).unwrap();
    assert_eq!(dest, ndarray::arr3(&[[[42.; 4]; 3]; 2]));
}

#[test]
fn test_try_into_array3_fail() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());
    for n in [0, 1, 2, 4, 5, 6, 7, 8] {
        let src = Node::fill(&g, &hw, Shape::from_slice(&vec![1; n]), 42.);
        assert!(ndarray::Array3::<f32>::try_from(src).is_err());
    }
}

#[test]
fn test_try_into_array4_0x0x0x0() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());
    let src = Node::fill(&g, &hw, Shape::new([0, 0, 0, 0]), 42.);
    let dest = ndarray::Array4::<f32>::try_from(src).unwrap();
    assert_eq!(
        dest,
        ndarray::Array4::<f32>::from_shape_vec((0, 0, 0, 0), vec![]).unwrap()
    );
}

#[test]
fn test_try_into_array4_2x3x4x5() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());
    let src = Node::fill(&g, &hw, Shape::new([2, 3, 4, 5]), 42.);
    let dest = ndarray::Array4::<f32>::try_from(src).unwrap();
    assert_eq!(
        dest,
        ndarray::Array4::<f32>::from_shape_vec((2, 3, 4, 5), vec![42.; 2 * 3 * 4 * 5]).unwrap()
    );
}

#[test]
fn test_try_into_array4_fail() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());
    for n in [0, 1, 2, 3, 5, 6, 7, 8] {
        let src = Node::fill(&g, &hw, Shape::from_slice(&vec![1; n]), 42.);
        assert!(ndarray::Array4::<f32>::try_from(src).is_err());
    }
}

#[test]
fn test_try_into_array5_0x0x0x0x0() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());
    let src = Node::fill(&g, &hw, Shape::new([0, 0, 0, 0, 0]), 42.);
    let dest = ndarray::Array5::<f32>::try_from(src).unwrap();
    assert_eq!(
        dest,
        ndarray::Array5::<f32>::from_shape_vec((0, 0, 0, 0, 0), vec![]).unwrap()
    );
}

#[test]
fn test_try_into_array5_2x3x4x5x6() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());
    let src = Node::fill(&g, &hw, Shape::new([2, 3, 4, 5, 6]), 42.);
    let dest = ndarray::Array5::<f32>::try_from(src).unwrap();
    assert_eq!(
        dest,
        ndarray::Array5::<f32>::from_shape_vec((2, 3, 4, 5, 6), vec![42.; 2 * 3 * 4 * 5 * 6])
            .unwrap()
    );
}

#[test]
fn test_try_into_array5_fail() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());
    for n in [0, 1, 2, 3, 4, 6, 7, 8] {
        let src = Node::fill(&g, &hw, Shape::from_slice(&vec![1; n]), 42.);
        assert!(ndarray::Array5::<f32>::try_from(src).is_err());
    }
}

#[test]
fn test_try_into_array6_0x0x0x0x0x0() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());
    let src = Node::fill(&g, &hw, Shape::new([0, 0, 0, 0, 0, 0]), 42.);
    let dest = ndarray::Array6::<f32>::try_from(src).unwrap();
    assert_eq!(
        dest,
        ndarray::Array6::<f32>::from_shape_vec((0, 0, 0, 0, 0, 0), vec![]).unwrap()
    );
}

#[test]
fn test_try_into_array6_2x3x4x5x6x7() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());
    let src = Node::fill(&g, &hw, Shape::new([2, 3, 4, 5, 6, 7]), 42.);
    let dest = ndarray::Array6::<f32>::try_from(src).unwrap();
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
fn test_try_into_array6_fail() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());
    for n in [0, 1, 2, 3, 4, 5, 7, 8] {
        let src = Node::fill(&g, &hw, Shape::from_slice(&vec![1; n]), 42.);
        assert!(ndarray::Array6::<f32>::try_from(src).is_err());
    }
}

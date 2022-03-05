use crate::hardware::cpu::CpuHardware;
use crate::node::*;

#[test]
fn test_steps() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());
    let lhs = Node::from_scalar(&g, &hw, 1.);
    let rhs = Node::from_scalar(&g, &hw, 2.);
    let ret = lhs + rhs;

    assert_eq!(lhs, Node::new(&g, 0));
    assert_eq!(rhs, Node::new(&g, 1));
    assert_eq!(ret, Node::new(&g, 2));
    assert_eq!(lhs.shape(), Shape::new([]));
    assert_eq!(rhs.shape(), Shape::new([]));
    assert_eq!(ret.shape(), Shape::new([]));
    assert!(ptr::eq(lhs.hardware(), &hw));
    assert!(ptr::eq(rhs.hardware(), &hw));
    assert!(ptr::eq(ret.hardware(), &hw));

    {
        let g = g.borrow();
        assert_eq!(g.num_steps(), 3);
        assert_eq!(g.get_step(0).unwrap().operator.name(), "Constant");
        assert_eq!(g.get_step(1).unwrap().operator.name(), "Constant");
        assert_eq!(g.get_step(2).unwrap().operator.name(), "Add");
    }

    let retval = ret.calculate();
    assert_eq!(*retval.shape(), Shape::new([]));
    assert_eq!(retval.get_scalar_f32(), Ok(3.));
}

#[test]
fn test_neg() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());

    let src = Node::from_scalar(&g, &hw, 42.);
    let dest = -src;

    assert_eq!(src.shape(), Shape::new([]));
    assert_eq!(dest.shape(), Shape::new([]));
    assert!(ptr::eq(src.hardware(), &hw));
    assert!(ptr::eq(dest.hardware(), &hw));

    assert_eq!(dest.calculate().get_scalar_f32(), Ok(-42.));
}

#[test]
fn test_add() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());

    let lhs = Node::from_scalar(&g, &hw, 1.);
    let rhs = Node::from_scalar(&g, &hw, 2.);
    let ret = lhs + rhs;

    assert_eq!(lhs.shape(), Shape::new([]));
    assert_eq!(rhs.shape(), Shape::new([]));
    assert_eq!(ret.shape(), Shape::new([]));
    assert!(ptr::eq(lhs.hardware(), &hw));
    assert!(ptr::eq(rhs.hardware(), &hw));
    assert!(ptr::eq(ret.hardware(), &hw));

    assert_eq!(ret.calculate().get_scalar_f32(), Ok(3.));
}

#[test]
fn test_sub() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());

    let lhs = Node::from_scalar(&g, &hw, 1.);
    let rhs = Node::from_scalar(&g, &hw, 2.);
    let ret = lhs - rhs;

    assert_eq!(lhs.shape(), Shape::new([]));
    assert_eq!(rhs.shape(), Shape::new([]));
    assert_eq!(ret.shape(), Shape::new([]));
    assert!(ptr::eq(lhs.hardware(), &hw));
    assert!(ptr::eq(rhs.hardware(), &hw));
    assert!(ptr::eq(ret.hardware(), &hw));

    assert_eq!(ret.calculate().get_scalar_f32(), Ok(-1.));
}

#[test]
fn test_mul() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());

    let lhs = Node::from_scalar(&g, &hw, 1.);
    let rhs = Node::from_scalar(&g, &hw, 2.);
    let ret = lhs * rhs;

    assert_eq!(lhs.shape(), Shape::new([]));
    assert_eq!(rhs.shape(), Shape::new([]));
    assert_eq!(ret.shape(), Shape::new([]));
    assert!(ptr::eq(lhs.hardware(), &hw));
    assert!(ptr::eq(rhs.hardware(), &hw));
    assert!(ptr::eq(ret.hardware(), &hw));

    assert_eq!(f32::try_from(ret), Ok(2.));
}

#[test]
fn test_div() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());

    let lhs = Node::from_scalar(&g, &hw, 1.);
    let rhs = Node::from_scalar(&g, &hw, 2.);
    let ret = lhs / rhs;

    assert_eq!(lhs.shape(), Shape::new([]));
    assert_eq!(rhs.shape(), Shape::new([]));
    assert_eq!(ret.shape(), Shape::new([]));
    assert!(ptr::eq(lhs.hardware(), &hw));
    assert!(ptr::eq(rhs.hardware(), &hw));
    assert!(ptr::eq(ret.hardware(), &hw));

    assert_eq!(f32::try_from(ret), Ok(0.5));
}

#[test]
fn test_fill_scalar() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());
    let ret = Node::fill(&g, &hw, Shape::new([]), 123.);
    assert_eq!(ret.shape(), Shape::new([]));
    assert!(ptr::eq(ret.hardware(), &hw));
    assert_eq!(f32::try_from(ret), Ok(123.));
}

#[test]
fn test_fill_0() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());
    let ret = Node::fill(&g, &hw, Shape::new([0]), 123.);
    assert_eq!(ret.shape(), Shape::new([0]));
    assert!(ptr::eq(ret.hardware(), &hw));
    assert_eq!(ret.calculate().get_values_f32(), vec![]);
}

#[test]
fn test_fill_n() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());
    let ret = Node::fill(&g, &hw, Shape::new([3]), 123.);
    assert_eq!(ret.shape(), Shape::new([3]));
    assert!(ptr::eq(ret.hardware(), &hw));
    assert_eq!(ret.calculate().get_values_f32(), vec![123., 123., 123.]);
}

#[test]
fn test_multiple_computation() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());

    let a = Node::from_scalar(&g, &hw, 1.);
    let b = Node::from_scalar(&g, &hw, 2.);
    let c = Node::from_scalar(&g, &hw, 3.);
    let y = a + -b * c;

    assert_eq!(a.shape(), Shape::new([]));
    assert_eq!(b.shape(), Shape::new([]));
    assert_eq!(c.shape(), Shape::new([]));
    assert_eq!(y.shape(), Shape::new([]));
    assert!(ptr::eq(a.hardware(), &hw));
    assert!(ptr::eq(b.hardware(), &hw));
    assert!(ptr::eq(c.hardware(), &hw));
    assert!(ptr::eq(y.hardware(), &hw));

    assert_eq!(f32::try_from(y), Ok(-5.));
}

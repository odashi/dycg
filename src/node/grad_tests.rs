use crate::hardware::cpu::CpuHardware;
use crate::node::*;

#[test]
fn test_empty() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());

    let x = 42f32.into_node(&g, &hw);

    let gx = grad(x, &[]);
    assert!(gx.is_empty());
}

#[test]
fn test_self() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());

    let x = 42f32.into_node(&g, &hw);

    let gx = grad(x, &[x]);
    assert_eq!(gx.len(), 1);

    assert_eq!(gx[0].shape(), Shape::new([]));
    assert!(ptr::eq(gx[0].hardware(), &hw));

    // dx/dx == 1
    assert_eq!(f32::try_from(gx[0]), Ok(1.));
}

#[test]
fn test_unrelated() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());

    let x = 42f32.into_node(&g, &hw);
    let y = 42f32.into_node(&g, &hw);

    let gx = grad(y, &[x]);
    assert_eq!(gx.len(), 1);

    assert_eq!(gx[0].shape(), Shape::new([]));
    assert!(ptr::eq(gx[0].hardware(), &hw));

    // dy/dx == 0 since y is not calculated by x.
    assert_eq!(f32::try_from(gx[0]), Ok(0.));
}

#[test]
fn test_neg() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());

    let x = 42f32.into_node(&g, &hw);
    let y = -x;

    let gx = grad(y, &[x]);
    assert_eq!(gx.len(), 1);

    assert_eq!(gx[0].shape(), Shape::new([]));
    assert!(ptr::eq(gx[0].hardware(), &hw));

    // dy/dx == -1
    assert_eq!(f32::try_from(gx[0]), Ok(-1.));
}

#[test]
fn test_add() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());

    let a = 123f32.into_node(&g, &hw);
    let b = 456f32.into_node(&g, &hw);
    let y = a + b;

    let gx = grad(y, &[a, b]);
    assert_eq!(gx.len(), 2);

    assert_eq!(gx[0].shape(), Shape::new([]));
    assert_eq!(gx[1].shape(), Shape::new([]));
    assert!(ptr::eq(gx[0].hardware(), &hw));
    assert!(ptr::eq(gx[1].hardware(), &hw));

    // dy/da == 1
    assert_eq!(f32::try_from(gx[0]), Ok(1.));
    // dy/db == 1
    assert_eq!(f32::try_from(gx[1]), Ok(1.));
}

#[test]
fn test_sub() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());

    let a = 123f32.into_node(&g, &hw);
    let b = 456f32.into_node(&g, &hw);
    let y = a - b;

    let gx = grad(y, &[a, b]);
    assert_eq!(gx.len(), 2);

    assert_eq!(gx[0].shape(), Shape::new([]));
    assert_eq!(gx[1].shape(), Shape::new([]));
    assert!(ptr::eq(gx[0].hardware(), &hw));
    assert!(ptr::eq(gx[1].hardware(), &hw));

    // dy/da == 1
    assert_eq!(f32::try_from(gx[0]), Ok(1.));
    // dy/db == -1
    assert_eq!(f32::try_from(gx[1]), Ok(-1.));
}

#[test]
fn test_mul() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());

    let a = 123f32.into_node(&g, &hw);
    let b = 456f32.into_node(&g, &hw);
    let y = a * b;

    let gx = grad(y, &[a, b]);
    assert_eq!(gx.len(), 2);

    assert_eq!(gx[0].shape(), Shape::new([]));
    assert_eq!(gx[1].shape(), Shape::new([]));
    assert!(ptr::eq(gx[0].hardware(), &hw));
    assert!(ptr::eq(gx[1].hardware(), &hw));

    // dy/da == b
    assert_eq!(f32::try_from(gx[0]), Ok(456.));
    // dy/db == a
    assert_eq!(f32::try_from(gx[1]), Ok(123.));
}

#[test]
fn test_mul_quadratic() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());

    let x = 123f32.into_node(&g, &hw);
    // This calculation generates a diamond dependency between x and y
    // so that gradient summation x + x is happened during backpropagation.
    let y = x * x;

    let gx = grad(y, &[x]);
    assert_eq!(gx.len(), 1);

    assert_eq!(gx[0].shape(), Shape::new([]));
    assert!(ptr::eq(gx[0].hardware(), &hw));

    // dy/dx == 2x, internally calculated by x + x.
    assert_eq!(f32::try_from(gx[0]), Ok(246.));
}

#[test]
fn test_div() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());

    let a = 3f32.into_node(&g, &hw);
    let b = 2f32.into_node(&g, &hw);
    let y = a / b;

    let gx = grad(y, &[a, b]);
    assert_eq!(gx.len(), 2);

    assert_eq!(gx[0].shape(), Shape::new([]));
    assert_eq!(gx[1].shape(), Shape::new([]));
    assert!(ptr::eq(gx[0].hardware(), &hw));
    assert!(ptr::eq(gx[1].hardware(), &hw));

    // dy/da == 1/b
    assert_eq!(f32::try_from(gx[0]), Ok(0.5));
    // dy/db == -a/b^2
    assert_eq!(f32::try_from(gx[1]), Ok(-0.75));
}

#[test]
fn test_multiple_computation() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());

    let a = 1f32.into_node(&g, &hw);
    let b = 2f32.into_node(&g, &hw);
    let c = 3f32.into_node(&g, &hw);
    let y = a + -b * c;

    let gx = grad(y, &[a, b, c]);
    assert_eq!(gx.len(), 3);

    assert_eq!(gx[0].shape(), Shape::new([]));
    assert_eq!(gx[1].shape(), Shape::new([]));
    assert_eq!(gx[2].shape(), Shape::new([]));
    assert!(ptr::eq(gx[0].hardware(), &hw));
    assert!(ptr::eq(gx[1].hardware(), &hw));
    assert!(ptr::eq(gx[2].hardware(), &hw));

    // dy/da == 1
    assert_eq!(f32::try_from(gx[0]), Ok(1.));
    // dy/db == -c
    assert_eq!(f32::try_from(gx[1]), Ok(-3.));
    // dy/dc = -b
    assert_eq!(f32::try_from(gx[2]), Ok(-2.));
}

#[test]
fn test_higher_order_gradients() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());

    let x = 5f32.into_node(&g, &hw);
    let y = x * x * x;

    let gx1 = grad(y, &[x])[0];
    let gx2 = grad(gx1, &[x])[0];
    let gx3 = grad(gx2, &[x])[0];
    let gx4 = grad(gx3, &[x])[0];

    assert_eq!(gx1.shape(), Shape::new([]));
    assert_eq!(gx2.shape(), Shape::new([]));
    assert_eq!(gx3.shape(), Shape::new([]));
    assert_eq!(gx4.shape(), Shape::new([]));
    assert!(ptr::eq(gx1.hardware(), &hw));
    assert!(ptr::eq(gx2.hardware(), &hw));
    assert!(ptr::eq(gx3.hardware(), &hw));
    assert!(ptr::eq(gx4.hardware(), &hw));

    // y' == dy/dx == 3x^2
    assert_eq!(f32::try_from(gx1), Ok(75.));
    // y'' == 6x
    assert_eq!(f32::try_from(gx2), Ok(30.));
    // y''' == 6
    assert_eq!(f32::try_from(gx3), Ok(6.));
    // y'''' == 0
    assert_eq!(f32::try_from(gx4), Ok(0.));
}

#[test]
fn test_gradient_of_multiple_variables() {
    let hw = RefCell::new(CpuHardware::new());
    let g = RefCell::new(Graph::new());

    let a = 2f32.into_node(&g, &hw);
    let b = 3f32.into_node(&g, &hw);
    let y = a * a * b;

    let y_a = grad(y, &[a])[0];
    let y_b = grad(y, &[b])[0];

    let y_aa = grad(y_a, &[a])[0];
    let y_ab = grad(y_a, &[b])[0];
    let y_ba = grad(y_b, &[a])[0];
    let y_bb = grad(y_b, &[b])[0];

    let y_aaa = grad(y_aa, &[a])[0];
    let y_aab = grad(y_aa, &[b])[0];
    let y_aba = grad(y_ab, &[a])[0];
    let y_abb = grad(y_ab, &[b])[0];
    let y_baa = grad(y_ba, &[a])[0];
    let y_bab = grad(y_ba, &[b])[0];
    let y_bba = grad(y_bb, &[a])[0];
    let y_bbb = grad(y_bb, &[b])[0];

    assert_eq!(f32::try_from(y_a), Ok(12.)); // 2ab
    assert_eq!(f32::try_from(y_b), Ok(4.)); // a^2

    assert_eq!(f32::try_from(y_aa), Ok(6.)); // 2b
    assert_eq!(f32::try_from(y_ab), Ok(4.)); // 2a
    assert_eq!(f32::try_from(y_ba), Ok(4.)); // 2a
    assert_eq!(f32::try_from(y_bb), Ok(0.)); // 0

    assert_eq!(f32::try_from(y_aaa), Ok(0.)); // 0
    assert_eq!(f32::try_from(y_aab), Ok(2.)); // 2
    assert_eq!(f32::try_from(y_aba), Ok(2.)); // 2
    assert_eq!(f32::try_from(y_abb), Ok(0.)); // 0
    assert_eq!(f32::try_from(y_baa), Ok(2.)); // 2
    assert_eq!(f32::try_from(y_bab), Ok(0.)); // 0
    assert_eq!(f32::try_from(y_bba), Ok(0.)); // 0
    assert_eq!(f32::try_from(y_bbb), Ok(0.)); // 0
}

#[test]
#[should_panic]
fn test_different_graph() {
    let hw = RefCell::new(CpuHardware::new());
    let g1 = RefCell::new(Graph::new());
    let g2 = RefCell::new(Graph::new());

    let x = 1f32.into_node(&g1, &hw);
    let y = 2f32.into_node(&g2, &hw);
    let _gx = grad(y, &[x])[0];
}

use crate::operator::*;

pub(crate) struct Mul;

impl Mul {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl<'hw> Operator<'hw> for Mul {
    fn name(&self) -> String {
        String::from("Mul")
    }

    fn input_size(&self) -> usize {
        2
    }

    fn perform_shape(&self, inputs: &[&Shape]) -> Result<Shape> {
        inputs[0].elementwise(inputs[1])
    }

    fn perform(&self, inputs: &[&Array<'hw>]) -> Result<Array<'hw>> {
        inputs[0].elementwise_mul_f32(inputs[1])
    }

    fn get_gradient_fn(&self) -> Option<Box<dyn Gradient>> {
        Some(Box::new(MulGrad {}))
    }
}

/// Gradient for Mul.
struct MulGrad;

impl Gradient for MulGrad {
    fn perform<'hw: 'op, 'op: 'g, 'g>(
        &self,
        x: &[Node<'hw, 'op, 'g>],
        _y: Node<'hw, 'op, 'g>,
        gy: Node<'hw, 'op, 'g>,
    ) -> Vec<Node<'hw, 'op, 'g>> {
        vec![gy * x[1], gy * x[0]]
    }
}

#[cfg(test)]
mod tests {
    use crate::array::IntoArray;
    use crate::hardware::cpu::CpuHardware;
    use crate::operator::mul::*;

    #[test]
    fn test_properties() {
        let op = Mul::new();
        assert_eq!(op.name(), "Mul");
        assert_eq!(op.input_size(), 2);
    }

    #[rustfmt::skip]
    #[test]
    fn test_perform_shape() {
        let op = Mul::new();
        assert_eq!(op.perform_shape(&[&Shape::new([]), &Shape::new([])]), Ok(Shape::new([])));
        assert_eq!(op.perform_shape(&[&Shape::new([0]), &Shape::new([0])]), Ok(Shape::new([0])));
        assert_eq!(op.perform_shape(&[&Shape::new([3]), &Shape::new([3])]), Ok(Shape::new([3])));
    }

    #[rustfmt::skip]
    #[test]
    fn test_perform_shape_invalid() {
        let op = Mul::new();
        assert!(op.perform_shape(&[&Shape::new([]), &Shape::new([0])]).is_err());
        assert!(op.perform_shape(&[&Shape::new([]), &Shape::new([3])]).is_err());
        assert!(op.perform_shape(&[&Shape::new([0]), &Shape::new([])]).is_err());
        assert!(op.perform_shape(&[&Shape::new([0]), &Shape::new([3])]).is_err());
        assert!(op.perform_shape(&[&Shape::new([3]), &Shape::new([])]).is_err());
        assert!(op.perform_shape(&[&Shape::new([3]), &Shape::new([0])]).is_err());
    }

    #[test]
    fn test_perform_hardware() {
        let hw1 = RefCell::new(CpuHardware::new());
        let hw2 = RefCell::new(CpuHardware::new());
        let op = Mul::new();

        assert!(ptr::eq(op.perform_hardware(&[&hw1, &hw1]).unwrap(), &hw1));
        assert!(ptr::eq(op.perform_hardware(&[&hw2, &hw2]).unwrap(), &hw2));
        assert!(op.perform_hardware(&[&hw1, &hw2]).is_err());
    }

    #[test]
    fn test_perform() {
        let hw = RefCell::new(CpuHardware::new());
        let op = Mul::new();
        let lhs = 1f32.into_array(&hw);
        let rhs = 2f32.into_array(&hw);
        let expected = 2f32.into_array(&hw);
        let observed = op.perform(&[&lhs, &rhs]).unwrap();
        assert_eq!(observed.shape(), expected.shape());
        assert_eq!(observed.get_scalar_f32(), expected.get_scalar_f32());
    }
}

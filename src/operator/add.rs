use crate::operator::*;

pub(crate) struct Add;

impl Add {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl<'hw> Operator<'hw> for Add {
    fn name(&self) -> String {
        String::from("Add")
    }

    fn input_size(&self) -> usize {
        2
    }

    fn perform_shape(&self, inputs: &[&Shape]) -> Result<Shape> {
        inputs[0].elementwise(inputs[1])
    }

    fn perform(&self, inputs: &[&Array<'hw>]) -> Result<Array<'hw>> {
        inputs[0].elementwise_add_f32(inputs[1])
    }

    fn get_gradient_fn(&self) -> Option<Box<dyn Gradient>> {
        Some(Box::new(AddGrad {}))
    }
}

/// Gradient for Add.
struct AddGrad;

impl Gradient for AddGrad {
    fn perform<'hw: 'op, 'op: 'g, 'g>(
        &self,
        _x: &[Node<'hw, 'op, 'g>],
        _y: Node<'hw, 'op, 'g>,
        gy: Node<'hw, 'op, 'g>,
    ) -> Vec<Node<'hw, 'op, 'g>> {
        vec![gy, gy]
    }
}

#[cfg(test)]
mod tests {
    use crate::array::IntoArray;
    use crate::hardware::cpu::CpuHardware;
    use crate::operator::add::*;

    #[test]
    fn test_properties() {
        let op = Add::new();
        assert_eq!(op.name(), "Add");
        assert_eq!(op.input_size(), 2);
    }

    #[rustfmt::skip]
    #[test]
    fn test_perform_shape() {
        let op = Add::new();
        assert_eq!(op.perform_shape(&[&Shape::new([]), &Shape::new([])]), Ok(Shape::new([])));
        assert_eq!(op.perform_shape(&[&Shape::new([0]), &Shape::new([0])]), Ok(Shape::new([0])));
        assert_eq!(op.perform_shape(&[&Shape::new([3]), &Shape::new([3])]), Ok(Shape::new([3])));
    }

    #[rustfmt::skip]
    #[test]
    fn test_perform_shape_invalid() {
        let op = Add::new();
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
        let op = Add::new();

        assert!(ptr::eq(op.perform_hardware(&[&hw1, &hw1]).unwrap(), &hw1));
        assert!(ptr::eq(op.perform_hardware(&[&hw2, &hw2]).unwrap(), &hw2));
        assert!(op.perform_hardware(&[&hw1, &hw2]).is_err());
    }

    #[test]
    fn test_perform() {
        let hw = RefCell::new(CpuHardware::new());
        let op = Add::new();
        let lhs = 1f32.into_array(&hw);
        let rhs = 2f32.into_array(&hw);
        let expected = 3f32.into_array(&hw);
        let observed = op.perform(&[&lhs, &rhs]).unwrap();
        assert_eq!(observed.shape(), expected.shape());
        assert_eq!(observed.get_scalar_f32(), expected.get_scalar_f32());
    }
}

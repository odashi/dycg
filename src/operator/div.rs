use crate::operator::*;

pub(crate) struct Div;

impl Div {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl<'hw> Operator<'hw> for Div {
    fn name(&self) -> String {
        String::from("Div")
    }

    fn input_size(&self) -> usize {
        2
    }

    fn perform_shape(&self, inputs: &[&Shape]) -> Result<Shape> {
        inputs[0].elementwise(inputs[1])
    }

    fn perform(&self, inputs: &[&Array<'hw>]) -> Result<Array<'hw>> {
        inputs[0].elementwise_div_f32(inputs[1])
    }

    fn gradient<'op: 'g, 'g>(
        &self,
        x: &[Node<'hw, 'op, 'g>],
        y: Node<'hw, 'op, 'g>,
        gy: Node<'hw, 'op, 'g>,
    ) -> Result<Vec<Node<'hw, 'op, 'g>>>
    where
        'hw: 'op,
    {
        let gx0 = gy / x[1];
        Ok(vec![gx0, -y * gx0])
    }
}

#[cfg(test)]
mod tests {
    use crate::hardware::cpu::CpuHardware;
    use crate::operator::div::*;

    #[test]
    fn test_properties() {
        let op = Div::new();
        assert_eq!(op.name(), "Div");
        assert_eq!(op.input_size(), 2);
    }

    #[rustfmt::skip]
    #[test]
    fn test_perform_shape() {
        let op = Div::new();
        assert_eq!(op.perform_shape(&[&Shape::new([]), &Shape::new([])]), Ok(Shape::new([])));
        assert_eq!(op.perform_shape(&[&Shape::new([0]), &Shape::new([0])]), Ok(Shape::new([0])));
        assert_eq!(op.perform_shape(&[&Shape::new([3]), &Shape::new([3])]), Ok(Shape::new([3])));
    }

    #[rustfmt::skip]
    #[test]
    fn test_perform_shape_invalid() {
        let op = Div::new();
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
        let op = Div::new();

        assert!(ptr::eq(op.perform_hardware(&[&hw1, &hw1]).unwrap(), &hw1));
        assert!(ptr::eq(op.perform_hardware(&[&hw2, &hw2]).unwrap(), &hw2));
        assert!(op.perform_hardware(&[&hw1, &hw2]).is_err());
    }

    #[test]
    fn test_perform() {
        let hw = RefCell::new(CpuHardware::new());
        let op = Div::new();
        let lhs = Array::scalar_f32(&hw, 1.);
        let rhs = Array::scalar_f32(&hw, 2.);
        let expected = Array::scalar_f32(&hw, 0.5);
        let observed = op.perform(&[&lhs, &rhs]).unwrap();
        assert_eq!(observed.shape(), expected.shape());
        assert_eq!(observed.get_scalar_f32(), expected.get_scalar_f32());
    }
}

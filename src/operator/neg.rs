use crate::operator::*;

pub(crate) struct Neg;

impl Neg {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl<'hw> Operator<'hw> for Neg {
    fn name(&self) -> String {
        String::from("Neg")
    }

    fn input_size(&self) -> usize {
        1
    }

    fn perform_shape(&self, inputs: &[&Shape]) -> Result<Shape> {
        Ok(inputs[0].clone())
    }

    fn perform(&self, inputs: &[&Array<'hw>]) -> Result<Array<'hw>> {
        Ok(inputs[0].elementwise_neg_f32())
    }

    fn get_gradient_fn(&self) -> Option<Box<dyn Gradient>> {
        Some(Box::new(NegGrad {}))
    }
}

/// Gradient for Neg.
struct NegGrad;

impl Gradient for NegGrad {
    fn perform<'hw: 'op, 'op: 'g, 'g>(
        &self,
        _x: &[Node<'hw, 'op, 'g>],
        _y: Node<'hw, 'op, 'g>,
        gy: Node<'hw, 'op, 'g>,
    ) -> Vec<Node<'hw, 'op, 'g>> {
        vec![-gy]
    }
}

#[cfg(test)]
mod tests {
    use crate::array::IntoArray;
    use crate::hardware::cpu::CpuHardware;
    use crate::operator::neg::*;

    #[test]
    fn test_properties() {
        let op = Neg::new();
        assert_eq!(op.name(), "Neg");
        assert_eq!(op.input_size(), 1);
    }

    #[rustfmt::skip]
    #[test]
    fn test_perform_shape() {
        let op = Neg::new();
        
        assert_eq!(op.perform_shape(&[&Shape::new([])]), Ok(Shape::new([])));
        assert_eq!(op.perform_shape(&[&Shape::new([0])]), Ok(Shape::new([0])));
        assert_eq!(op.perform_shape(&[&Shape::new([3])]), Ok(Shape::new([3])));
    }

    #[test]
    fn test_perform_hardware() {
        let hw = RefCell::new(CpuHardware::new());
        let op = Neg::new();

        assert!(ptr::eq(op.perform_hardware(&[&hw]).unwrap(), &hw));
    }

    #[test]
    fn test_perform() {
        let hw = RefCell::new(CpuHardware::new());
        let op = Neg::new();
        let input = 42f32.into_array(&hw);
        let expected = (-42f32).into_array(&hw);
        let observed = op.perform(&[&input]).unwrap();
        assert_eq!(observed.shape(), expected.shape());
        assert_eq!(observed.get_scalar_f32(), expected.get_scalar_f32());
    }
}

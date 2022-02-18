use crate::array::Array;
use crate::node::Node;
use crate::operator::Operator;
use crate::result::Result;
use crate::shape::Shape;

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

    fn gradient<'op: 'g, 'g>(
        &self,
        x: &[Node<'hw, 'op, 'g>],
        _y: Node<'hw, 'op, 'g>,
        gy: Node<'hw, 'op, 'g>,
    ) -> Result<Vec<Node<'hw, 'op, 'g>>>
    where
        'hw: 'op,
    {
        Ok(vec![gy * x[1], gy * x[0]])
    }
}

#[cfg(test)]
mod tests {
    use crate::array::Array;
    use crate::hardware::cpu::CpuHardware;
    use crate::operator::mul::Mul;
    use crate::operator::Operator;
    use std::cell::RefCell;

    #[test]
    fn test_mul_op() {
        let hw = RefCell::new(CpuHardware::new());
        let op = Mul::new();
        assert_eq!(op.name(), "Mul");
        assert_eq!(op.input_size(), 2);
        let inputs = vec![Array::new_scalar(&hw, 1.), Array::new_scalar(&hw, 2.)];
        let expected = Array::new_scalar(&hw, 2.);
        let observed = op.perform(&inputs.iter().collect::<Vec<_>>()).unwrap();
        assert_eq!(observed.shape(), expected.shape());
        assert_eq!(observed.to_scalar(), expected.to_scalar());
    }
}

use crate::array::Array;
use crate::node::Node;
use crate::operator::Operator;
use crate::result::Result;
use crate::shape::Shape;

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
    use crate::array::Array;
    use crate::hardware::cpu::CpuHardware;
    use crate::operator::div::Div;
    use crate::operator::Operator;
    use std::cell::RefCell;

    #[test]
    fn test_div_op() {
        let hw = RefCell::new(CpuHardware::new());
        let op = Div::new();
        assert_eq!(op.name(), "Div");
        assert_eq!(op.input_size(), 2);
        let inputs = vec![Array::new_scalar(&hw, 1.), Array::new_scalar(&hw, 2.)];
        let expected = Array::new_scalar(&hw, 0.5);
        let observed = op.perform(&inputs.iter().collect::<Vec<_>>()).unwrap();
        assert_eq!(observed.shape(), expected.shape());
        assert_eq!(observed.to_scalar(), expected.to_scalar());
    }
}

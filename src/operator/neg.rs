use crate::array::Array;
use crate::node::Node;
use crate::operator::Operator;
use crate::result::Result;

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

    fn perform(&self, inputs: &[&Array<'hw>]) -> Result<Array<'hw>> {
        Ok(inputs[0].elementwise_neg_f32()?)
    }

    fn gradient<'op, 'g>(
        &self,
        _x: &[Node<'hw, 'op, 'g>],
        _y: Node<'hw, 'op, 'g>,
        gy: Node<'hw, 'op, 'g>,
    ) -> Result<Vec<Node<'hw, 'op, 'g>>> {
        Ok(vec![-gy])
    }
}

#[cfg(test)]
mod tests {
    use crate::array::Array;
    use crate::hardware::cpu::CpuHardware;
    use crate::operator::neg::Neg;
    use crate::operator::Operator;
    use std::cell::RefCell;

    #[test]
    fn test_neg_op() {
        let hw = RefCell::new(CpuHardware::new());
        let op = Neg::new();
        assert_eq!(op.name(), "Neg");
        assert_eq!(op.input_size(), 1);
        let inputs = vec![Array::new_scalar(&hw, 42.)];
        let expected = Array::new_scalar(&hw, -42.);
        let observed = op.perform(&inputs.iter().collect::<Vec<_>>()).unwrap();
        assert_eq!(observed.shape(), expected.shape());
        assert_eq!(observed.to_scalar(), expected.to_scalar());
    }
}

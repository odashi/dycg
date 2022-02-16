use crate::array::Array;
use crate::node::Node;
use crate::operator::Operator;
use crate::result::Result;

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
    fn output_size(&self) -> usize {
        1
    }
    fn perform(&self, inputs: &[&Array<'hw>]) -> Result<Vec<Array<'hw>>> {
        Ok(vec![inputs[0].elementwise_add_f32(inputs[1])?])
    }

    fn gradient<'op, 'g>(
        &self,
        _x: &[&Node<'hw, 'op, 'g>],
        _y: &[&Node<'hw, 'op, 'g>],
        gy: &[&Node<'hw, 'op, 'g>],
    ) -> Result<Vec<Node<'hw, 'op, 'g>>> {
        Ok(vec![*gy[0], *gy[0]])
    }
}

#[cfg(test)]
mod tests {
    use crate::array::Array;
    use crate::hardware::cpu::CpuHardware;
    use crate::operator::add::Add;
    use crate::operator::Operator;
    use std::cell::RefCell;

    #[test]
    fn test_add_op() {
        let hw = RefCell::new(CpuHardware::new());
        let op = Add::new();
        assert_eq!(op.name(), "Add");
        assert_eq!(op.input_size(), 2);
        assert_eq!(op.output_size(), 1);
        let inputs = vec![Array::new_scalar(&hw, 1.), Array::new_scalar(&hw, 2.)];
        let expected = vec![Array::new_scalar(&hw, 3.)];
        let observed = op.perform(&inputs.iter().collect::<Vec<_>>()).unwrap();
        assert_eq!(observed.len(), expected.len());
        for i in 0..expected.len() {
            assert_eq!(observed[i].shape(), expected[i].shape());
            assert_eq!(observed[i].to_scalar(), expected[i].to_scalar());
        }
    }
}

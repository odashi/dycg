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

    fn perform(&self, inputs: &[&Array<'hw>]) -> Result<Array<'hw>> {
        Ok(inputs[0].elementwise_add_f32(inputs[1])?)
    }

    fn gradient<'op, 'g>(
        &self,
        _x: &[&Node<'hw, 'op, 'g>],
        _y: &Node<'hw, 'op, 'g>,
        gy: &Node<'hw, 'op, 'g>,
    ) -> Result<Vec<Node<'hw, 'op, 'g>>> {
        Ok(vec![*gy, *gy])
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
        let inputs = vec![Array::new_scalar(&hw, 1.), Array::new_scalar(&hw, 2.)];
        let expected = Array::new_scalar(&hw, 3.);
        let observed = op.perform(&inputs.iter().collect::<Vec<_>>()).unwrap();
        assert_eq!(observed.shape(), expected.shape());
        assert_eq!(observed.to_scalar(), expected.to_scalar());
    }
}

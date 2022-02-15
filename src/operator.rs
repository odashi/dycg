use crate::array::Array;
use crate::error::Error;
use crate::node::Node;
use crate::result::Result;

/// Operator represents an individual computation process in the computation graph.
pub(crate) trait Operator<'hw> {
    fn name(&self) -> String;
    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;
    fn perform(&self, inputs: &[&Array<'hw>]) -> Result<Vec<Array<'hw>>>;
    fn gradient<'op, 'g>(
        &self,
        _x: &[&Node<'hw, 'op, 'g>],
        _y: &[&Node<'hw, 'op, 'g>],
        _gy: &[&Node<'hw, 'op, 'g>],
    ) -> Result<Vec<Node<'hw, 'op, 'g>>> {
        Err(Error::NotSupported(format!(
            "No gradient definition for {}",
            self.name(),
        )))
    }
}

pub(crate) struct Constant<'hw> {
    value: Array<'hw>,
}

impl<'hw> Constant<'hw> {
    pub(crate) fn new(value: Array<'hw>) -> Self {
        Self { value }
    }
}

impl<'hw> Operator<'hw> for Constant<'hw> {
    fn name(&self) -> String {
        String::from("Constant")
    }

    fn input_size(&self) -> usize {
        0
    }

    fn output_size(&self) -> usize {
        1
    }

    fn perform(&self, _inputs: &[&Array<'hw>]) -> Result<Vec<Array<'hw>>> {
        Ok(vec![self.value.clone()])
    }
}

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

    fn output_size(&self) -> usize {
        1
    }

    fn perform(&self, inputs: &[&Array<'hw>]) -> Result<Vec<Array<'hw>>> {
        Ok(vec![inputs[0].elementwise_neg_f32()?])
    }

    fn gradient<'op, 'g>(
        &self,
        _x: &[&Node<'hw, 'op, 'g>],
        _y: &[&Node<'hw, 'op, 'g>],
        gy: &[&Node<'hw, 'op, 'g>],
    ) -> Result<Vec<Node<'hw, 'op, 'g>>> {
        Ok(vec![-*gy[0]])
    }
}

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

pub(crate) struct Sub;

impl Sub {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl<'hw> Operator<'hw> for Sub {
    fn name(&self) -> String {
        String::from("Sub")
    }
    fn input_size(&self) -> usize {
        2
    }
    fn output_size(&self) -> usize {
        1
    }
    fn perform(&self, inputs: &[&Array<'hw>]) -> Result<Vec<Array<'hw>>> {
        Ok(vec![inputs[0].elementwise_sub_f32(inputs[1])?])
    }

    fn gradient<'op, 'g>(
        &self,
        _x: &[&Node<'hw, 'op, 'g>],
        _y: &[&Node<'hw, 'op, 'g>],
        gy: &[&Node<'hw, 'op, 'g>],
    ) -> Result<Vec<Node<'hw, 'op, 'g>>> {
        Ok(vec![*gy[0], -*gy[0]])
    }
}

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
    fn output_size(&self) -> usize {
        1
    }
    fn perform(&self, inputs: &[&Array<'hw>]) -> Result<Vec<Array<'hw>>> {
        Ok(vec![inputs[0].elementwise_mul_f32(inputs[1])?])
    }

    fn gradient<'op, 'g>(
        &self,
        x: &[&Node<'hw, 'op, 'g>],
        _y: &[&Node<'hw, 'op, 'g>],
        gy: &[&Node<'hw, 'op, 'g>],
    ) -> Result<Vec<Node<'hw, 'op, 'g>>> {
        Ok(vec![*gy[0] * *x[1], *gy[0] * *x[0]])
    }
}

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
    fn output_size(&self) -> usize {
        1
    }
    fn perform(&self, inputs: &[&Array<'hw>]) -> Result<Vec<Array<'hw>>> {
        Ok(vec![inputs[0].elementwise_div_f32(inputs[1])?])
    }

    fn gradient<'op, 'g>(
        &self,
        x: &[&Node<'hw, 'op, 'g>],
        y: &[&Node<'hw, 'op, 'g>],
        gy: &[&Node<'hw, 'op, 'g>],
    ) -> Result<Vec<Node<'hw, 'op, 'g>>> {
        Ok(vec![*gy[0] / *x[1], -*gy[0] * *y[0] / *x[1]])
    }
}

#[cfg(test)]
mod tests {
    use crate::hardware::cpu::CpuHardware;
    use crate::operator::*;
    use std::cell::RefCell;

    #[test]
    fn test_constant_op() {
        let hw = RefCell::new(CpuHardware::new());
        let op = Constant::new(Array::new_scalar(&hw, 123.));
        assert_eq!(op.name(), "Constant");
        assert_eq!(op.input_size(), 0);
        assert_eq!(op.output_size(), 1);
        let input_refs = vec![];
        let expected = vec![Array::new_scalar(&hw, 123.)];
        let observed = op.perform(&input_refs).unwrap();
        assert_eq!(observed.len(), expected.len());
        for i in 0..expected.len() {
            assert_eq!(observed[i].shape(), expected[i].shape());
            assert_eq!(observed[i].to_scalar(), expected[i].to_scalar());
        }
    }

    #[test]
    pub fn test_add_op() {
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

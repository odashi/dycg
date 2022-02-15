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
        _outputs: &[&Node<'hw, 'op, 'g>],
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

macro_rules! define_elementwise_binary_op {
    ( $name:ident, $fn:ident ) => {
        pub(crate) struct $name;
        impl $name {
            pub(crate) fn new() -> Self {
                Self {}
            }
        }
        impl<'hw> Operator<'hw> for $name {
            fn name(&self) -> String {
                String::from(stringify!($name))
            }
            fn input_size(&self) -> usize {
                2
            }
            fn output_size(&self) -> usize {
                1
            }
            fn perform(&self, inputs: &[&Array<'hw>]) -> Result<Vec<Array<'hw>>> {
                Ok(vec![inputs[0].$fn(inputs[1])?])
            }
        }
    };
}

define_elementwise_binary_op!(Add, elementwise_add_f32);
define_elementwise_binary_op!(Sub, elementwise_sub_f32);
define_elementwise_binary_op!(Mul, elementwise_mul_f32);
define_elementwise_binary_op!(Div, elementwise_div_f32);

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

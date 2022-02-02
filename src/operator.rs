use crate::array::Array;
use crate::error::Error;
use crate::result::Result;

/// Operator represents an individual computation process in the computation graph.
pub(crate) trait Operator<'hw> {
    fn name(&self) -> String;
    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;
    fn perform(&self, inputs: &Vec<Array<'hw>>) -> Result<Vec<Array<'hw>>>;
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
    fn perform(&self, _inputs: &Vec<Array<'hw>>) -> Result<Vec<Array<'hw>>> {
        Ok(vec![self.value.clone()])
    }
}

macro_rules! define_binary_op {
    ( $name:ident, $lhs:ident, $rhs:ident, $impl:expr ) => {
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
            fn perform(&self, inputs: &Vec<Array<'hw>>) -> Result<Vec<Array<'hw>>> {
                let $lhs = inputs[0].clone();
                let $rhs = inputs[1].clone();
                Ok(vec![$impl])
            }
        }
    };
}

define_binary_op!(Add, lhs, rhs, lhs.elementwise_add_f32(&rhs)?);
define_binary_op!(Sub, lhs, rhs, lhs.elementwise_sub_f32(&rhs)?);
define_binary_op!(Mul, lhs, rhs, lhs.elementwise_mul_f32(&rhs)?);
define_binary_op!(Div, lhs, rhs, lhs.elementwise_div_f32(&rhs)?);

#[cfg(test)]
mod tests {
    use crate::hardware::HardwareMutex;
    use crate::{hardware::cpu::CpuHardware, operator::*};

    /// Helper function to make a hardware.
    fn make_hardware() -> HardwareMutex {
        HardwareMutex::new(Box::new(CpuHardware::new("test")))
    }

    #[test]
    fn test_constant_op() {
        let hw = make_hardware();
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
        let hw = make_hardware();
        let op = Add::new();
        assert_eq!(op.name(), "Add");
        assert_eq!(op.input_size(), 2);
        assert_eq!(op.output_size(), 1);
        let inputs = vec![Array::new_scalar(&hw, 1.), Array::new_scalar(&hw, 2.)];
        let expected = vec![Array::new_scalar(&hw, 3.)];
        let observed = op.perform(&inputs).unwrap();
        assert_eq!(observed.len(), expected.len());
        for i in 0..expected.len() {
            assert_eq!(observed[i].shape(), expected[i].shape());
            assert_eq!(observed[i].to_scalar(), expected[i].to_scalar());
        }
    }
}

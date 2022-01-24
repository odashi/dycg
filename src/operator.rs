use std::rc::Rc;

use crate::array::Array;
use crate::error::Error;
use crate::result::Result;

/// Operator represents an individual computation process in the computation graph.
pub(crate) trait Operator {
    fn name(&self) -> &str;
    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;
    fn perform(&self, inputs: &Vec<&Array>) -> Result<Vec<Rc<Array>>>;
}

pub(crate) struct Constant {
    value: Rc<Array>,
}

impl Constant {
    pub(crate) fn new(value: Array) -> Self {
        Self {
            value: Rc::new(value),
        }
    }
}

impl Operator for Constant {
    fn name(&self) -> &str {
        "Constant"
    }
    fn input_size(&self) -> usize {
        0
    }
    fn output_size(&self) -> usize {
        1
    }
    fn perform(&self, _inputs: &Vec<&Array>) -> Result<Vec<Rc<Array>>> {
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
        impl Operator for $name {
            fn name(&self) -> &str {
                stringify!($name)
            }
            fn input_size(&self) -> usize {
                2
            }
            fn output_size(&self) -> usize {
                1
            }
            fn perform(&self, inputs: &Vec<&Array>) -> Result<Vec<Rc<Array>>> {
                let $lhs = inputs
                    .get(0)
                    .ok_or(Error::OutOfRange(String::from("inputs[0] does not exist.")))?;
                let $rhs = inputs
                    .get(0)
                    .ok_or(Error::OutOfRange(String::from("inputs[1] does not exist.")))?;
                Ok(vec![Rc::new($impl)])
            }
        }
    };
}

define_binary_op!(Add, lhs, rhs, lhs.elementwise_add_f32(rhs)?);
define_binary_op!(Sub, lhs, rhs, lhs.elementwise_sub_f32(rhs)?);
define_binary_op!(Mul, lhs, rhs, lhs.elementwise_mul_f32(rhs)?);
define_binary_op!(Div, lhs, rhs, lhs.elementwise_div_f32(rhs)?);

#[cfg(test)]
mod tests {
    use crate::array::make_cpu_scalar;
    use crate::operator::*;

    #[test]
    fn test_constant_op() {
        let op = Constant::new(make_cpu_scalar(123.));
        assert_eq!(op.name(), "Constant");
        assert_eq!(op.input_size(), 0);
        assert_eq!(op.output_size(), 1);
        let input_refs = vec![];
        let expected = vec![make_cpu_scalar(123.)];
        let observed = op.perform(&input_refs).unwrap();
        assert_eq!(observed.len(), expected.len());
        for i in 0..expected.len() {
            assert_eq!(observed[i].shape(), expected[i].shape());
            assert_eq!(observed[i].into_scalar(), expected[i].into_scalar());
        }
    }

    #[test]
    pub fn test_add_op() {
        let op = Add::new();
        assert_eq!(op.name(), "Add");
        assert_eq!(op.input_size(), 2);
        assert_eq!(op.output_size(), 1);
        let inputs = vec![make_cpu_scalar(1.), make_cpu_scalar(2.)];
        let input_refs = inputs.iter().map(|x| x).collect::<Vec<_>>();
        let expected = vec![make_cpu_scalar(3.)];
        let observed = op.perform(&input_refs).unwrap();
        assert_eq!(observed.len(), expected.len());
        for i in 0..expected.len() {
            assert_eq!(observed[i].shape(), expected[i].shape());
            assert_eq!(observed[i].into_scalar(), expected[i].into_scalar());
        }
    }
}

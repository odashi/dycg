use std::rc::Rc;

use crate::array::Array;

/// Operator represents an individual computation process in the computation graph.
pub(crate) trait Operator {
    fn name(&self) -> &str;
    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;
    unsafe fn perform_unchecked(&self, inputs: &Vec<&Array>) -> Vec<Rc<Array>>;
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
    unsafe fn perform_unchecked(&self, _inputs: &Vec<&Array>) -> Vec<Rc<Array>> {
        vec![self.value.clone()]
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
            unsafe fn perform_unchecked(&self, inputs: &Vec<&Array>) -> Vec<Rc<Array>> {
                let $lhs = inputs.get_unchecked(0);
                let $rhs = inputs.get_unchecked(1);
                vec![Rc::new($impl)]
            }
        }
    };
}

macro_rules! define_elementwise_binary_op {
    ($name:ident, $op:expr) => {
        define_binary_op!(
            $name,
            lhs,
            rhs,
            Array::with_default_backend(
                *lhs.shape(),
                lhs.to_vec()
                    .iter()
                    .zip(rhs.to_vec().iter())
                    .map($op)
                    .collect(),
            )
            .unwrap()
        );
    };
}

define_elementwise_binary_op!(Add, |(a, b)| a + b);
define_elementwise_binary_op!(Sub, |(a, b)| a - b);
define_elementwise_binary_op!(Mul, |(a, b)| a * b);
define_elementwise_binary_op!(Div, |(a, b)| a / b);

#[cfg(test)]
mod tests {
    use crate::array::make_scalar;
    use crate::operator::*;

    #[test]
    fn test_constant_op() {
        let op = Constant::new(make_scalar(123.));
        assert_eq!(op.name(), "Constant");
        assert_eq!(op.input_size(), 0);
        assert_eq!(op.output_size(), 1);
        let input_refs = vec![];
        let expected = vec![make_scalar(123.)];
        let observed = unsafe { op.perform_unchecked(&input_refs) };
        assert_eq!(observed.len(), expected.len());
        for i in 0..expected.len() {
            assert_eq!(observed[i].shape(), expected[i].shape());
            assert_eq!(observed[i].to_vec(), expected[i].to_vec());
        }
    }

    #[test]
    pub fn test_add_op() {
        let op = Add::new();
        assert_eq!(op.name(), "Add");
        assert_eq!(op.input_size(), 2);
        assert_eq!(op.output_size(), 1);
        let inputs = vec![make_scalar(1.), make_scalar(2.)];
        let input_refs = inputs.iter().map(|x| x).collect::<Vec<_>>();
        let expected = vec![make_scalar(3.)];
        let observed = unsafe { op.perform_unchecked(&input_refs) };
        assert_eq!(observed.len(), expected.len());
        for i in 0..expected.len() {
            assert_eq!(observed[i].shape(), expected[i].shape());
            assert_eq!(observed[i].to_vec(), expected[i].to_vec());
        }
    }
}

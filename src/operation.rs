use crate::error::Error;
use crate::result::Result;
use crate::value::{make_scalar, Value};

/// Operation represents an individual computation process in the computation graph.
pub(crate) trait Operation {
    fn name(&self) -> &'static str;
    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;
    fn perform(&self, inputs: &Vec<&Value>) -> Result<Vec<Value>>;
}

macro_rules! define_binary_op {
    ( $name:ident, $arg1:ident, $arg2:ident, $impl:expr) => {
        pub struct $name;
        impl Operation for $name {
            fn name(&self) -> &'static str {
                stringify!($name)
            }
            fn input_size(&self) -> usize {
                2
            }
            fn output_size(&self) -> usize {
                1
            }
            fn perform(&self, inputs: &Vec<&Value>) -> Result<Vec<Value>> {
                let $arg1 = inputs[0];
                let $arg2 = inputs[1];
                if $arg1.shape() != $arg2.shape() {
                    return Err(Error::ShapeMismatched(format!(
                        "Binary operation of shapes {} and {} is not supported.",
                        $arg1.shape(),
                        $arg2.shape()
                    )));
                }
                Ok(vec![$impl])
            }
        }
    };
}

define_binary_op!(Add, a, b, {
    Value::new(
        *a.shape(),
        a.to_vec()
            .iter()
            .zip(b.to_vec().iter())
            .map(|(a, b)| a + b)
            .collect(),
    )
    .unwrap()
});

#[cfg(test)]
mod tests {
    use crate::operation::{Add, Operation};
    use crate::value::make_scalar;

    #[test]
    pub fn test_add_op() {
        assert_eq!(Add {}.name(), "Add");
        assert_eq!(Add {}.input_size(), 2);
        assert_eq!(Add {}.output_size(), 1);
        let inputs = vec![make_scalar(1.), make_scalar(2.)];
        let input_refs = inputs.iter().map(|x| x).collect::<Vec<_>>();
        let expected = vec![make_scalar(3.)];
        let observed = Add {}.perform(&input_refs).unwrap();
        assert_eq!(observed.len(), expected.len());
        for i in 0..expected.len() {
            assert_eq!(observed[i].shape(), expected[i].shape());
            assert_eq!(observed[i].to_vec(), expected[i].to_vec());
        }
    }
}

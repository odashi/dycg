use crate::array::Array;
use crate::operator::Operator;
use crate::result::Result;

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

#[cfg(test)]
mod tests {
    use crate::array::Array;
    use crate::hardware::cpu::CpuHardware;
    use crate::operator::constant::Constant;
    use crate::operator::Operator;
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
}

use crate::array::Array;
use crate::operator::Operator;
use crate::result::Result;
use crate::shape::Shape;

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

    fn perform_shape(&self, _inputs: &[&Shape]) -> Result<Shape> {
        Ok(self.value.shape().clone())
    }

    fn perform(&self, _inputs: &[&Array<'hw>]) -> Result<Array<'hw>> {
        Ok(self.value.clone())
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
        let input_refs = vec![];
        let expected = Array::new_scalar(&hw, 123.);
        let observed = op.perform(&input_refs).unwrap();
        assert_eq!(observed.shape(), expected.shape());
        assert_eq!(observed.to_scalar(), expected.to_scalar());
    }
}

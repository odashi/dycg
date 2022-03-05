use crate::operator::*;

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

    fn perform_hardware(
        &self,
        _inputs: &[&'hw RefCell<dyn Hardware>],
    ) -> Result<&'hw RefCell<dyn Hardware>> {
        Ok(self.value.hardware())
    }

    fn perform(&self, _inputs: &[&Array<'hw>]) -> Result<Array<'hw>> {
        Ok(self.value.clone())
    }
}

#[cfg(test)]
mod tests {
    use crate::array::IntoArray;
    use crate::hardware::cpu::CpuHardware;
    use crate::operator::constant::*;

    #[test]
    fn test_properties() {
        let hw = RefCell::new(CpuHardware::new());
        let op = Constant::new(123f32.into_array(&hw));
        assert_eq!(op.name(), "Constant");
        assert_eq!(op.input_size(), 0);
    }

    #[test]
    fn test_perform_shape_scalar() {
        let hw = RefCell::new(CpuHardware::new());
        let op = Constant::new(123f32.into_array(&hw));
        assert_eq!(op.perform_shape(&[]), Ok(Shape::new([])));
    }

    #[test]
    fn test_perform_shape_0() {
        let hw = RefCell::new(CpuHardware::new());
        let op = Constant::new(Array::fill_f32(&hw, Shape::new([0]), 123.));
        assert_eq!(op.perform_shape(&[]), Ok(Shape::new([0])));
    }

    #[test]
    fn test_perform_shape_n() {
        let hw = RefCell::new(CpuHardware::new());
        let op = Constant::new(Array::fill_f32(&hw, Shape::new([3]), 123.));
        assert_eq!(op.perform_shape(&[]), Ok(Shape::new([3])));
    }

    #[test]
    fn test_perform_hardware() {
        let hw = RefCell::new(CpuHardware::new());
        let op = Constant::new(123f32.into_array(&hw));
        assert!(ptr::eq(op.perform_hardware(&[]).unwrap(), &hw));
    }

    #[test]
    fn test_perform() {
        let hw = RefCell::new(CpuHardware::new());
        let op = Constant::new(123f32.into_array(&hw));
        let expected = 123f32.into_array(&hw);
        let observed = op.perform(&[]).unwrap();
        assert_eq!(observed.shape(), expected.shape());
        assert_eq!(observed.get_scalar_f32(), expected.get_scalar_f32());
    }
}

use crate::operator::*;

/// Fill operator: creates an array with specific hardware/shape, filled by a single value.
pub(crate) struct Fill<'hw> {
    /// Hardware to host the value.
    hardware: &'hw RefCell<dyn Hardware>,

    /// Shape of the resulting array.
    shape: Shape,

    /// Value of elements in the resulting array.
    value: f32,
}

impl<'hw> Fill<'hw> {
    pub(crate) fn new(hardware: &'hw RefCell<dyn Hardware>, shape: Shape, value: f32) -> Self {
        Self {
            hardware,
            shape,
            value,
        }
    }
}

impl<'hw> Operator<'hw> for Fill<'hw> {
    fn name(&self) -> String {
        String::from("Fill")
    }

    fn input_size(&self) -> usize {
        0
    }

    fn perform_shape(&self, _inputs: &[&Shape]) -> Result<Shape> {
        Ok(self.shape.clone())
    }

    fn perform_hardware(
        &self,
        _inputs: &[&'hw RefCell<dyn Hardware>],
    ) -> Result<&'hw RefCell<dyn Hardware>> {
        Ok(self.hardware)
    }

    fn perform(&self, _inputs: &[&Array<'hw>]) -> Result<Array<'hw>> {
        Ok(Array::fill_f32(
            self.hardware,
            self.shape.clone(),
            self.value,
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::hardware::cpu::CpuHardware;
    use crate::operator::fill::*;

    #[test]
    fn test_properties() {
        let hw = RefCell::new(CpuHardware::new());
        let op = Fill::new(&hw, Shape::new([]), 123.);
        assert_eq!(op.name(), "Fill");
        assert_eq!(op.input_size(), 0);
    }

    #[test]
    fn test_perform_shape_scalar() {
        let hw = RefCell::new(CpuHardware::new());
        let op = Fill::new(&hw, Shape::new([]), 123.);
        assert_eq!(op.perform_shape(&[]), Ok(Shape::new([])));
    }

    #[test]
    fn test_perform_shape_0() {
        let hw = RefCell::new(CpuHardware::new());
        let op = Fill::new(&hw, Shape::new([0]), 123.);
        assert_eq!(op.perform_shape(&[]), Ok(Shape::new([0])));
    }

    #[test]
    fn test_perform_shape_n() {
        let hw = RefCell::new(CpuHardware::new());
        let op = Fill::new(&hw, Shape::new([3]), 123.);
        assert_eq!(op.perform_shape(&[]), Ok(Shape::new([3])));
    }

    #[test]
    fn test_perform_hardware() {
        let hw = RefCell::new(CpuHardware::new());
        let op = Fill::new(&hw, Shape::new([]), 123.);
        assert!(ptr::eq(op.perform_hardware(&[]).unwrap(), &hw));
    }

    #[test]
    fn test_perform_scalar() {
        let hw = RefCell::new(CpuHardware::new());
        let op = Fill::new(&hw, Shape::new([]), 123.);
        let expected = Array::fill_f32(&hw, Shape::new([]), 123.);
        let observed = op.perform(&[]).unwrap();
        assert_eq!(observed.shape(), expected.shape());
        assert_eq!(observed.get_scalar_f32(), expected.get_scalar_f32());
    }

    #[test]
    fn test_perform_0() {
        let hw = RefCell::new(CpuHardware::new());
        let op = Fill::new(&hw, Shape::new([0]), 123.);
        let expected = Array::fill_f32(&hw, Shape::new([0]), 123.);
        let observed = op.perform(&[]).unwrap();
        assert_eq!(observed.shape(), expected.shape());
        assert_eq!(observed.get_values_f32(), expected.get_values_f32());
    }

    #[test]
    fn test_perform_n() {
        let hw = RefCell::new(CpuHardware::new());
        let op = Fill::new(&hw, Shape::new([3]), 123.);
        let expected = Array::fill_f32(&hw, Shape::new([3]), 123.);
        let observed = op.perform(&[]).unwrap();
        assert_eq!(observed.shape(), expected.shape());
        assert_eq!(observed.get_values_f32(), expected.get_values_f32());
    }
}

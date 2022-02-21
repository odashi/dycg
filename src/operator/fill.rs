use crate::array::Array;
use crate::hardware::Hardware;
use crate::operator::Operator;
use crate::result::Result;
use crate::shape::Shape;
use std::cell::RefCell;

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
        "Fill".to_string()
    }

    fn input_size(&self) -> usize {
        0
    }

    fn perform_shape(&self, _inputs: &[&Shape]) -> Result<Shape> {
        Ok(self.shape.clone())
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
    use crate::array::Array;
    use crate::hardware::cpu::CpuHardware;
    use crate::make_shape;
    use crate::operator::fill::Fill;
    use crate::operator::Operator;
    use std::cell::RefCell;

    #[test]
    fn test_op_scalar() {
        let hw = RefCell::new(CpuHardware::new());
        let op = Fill::new(&hw, make_shape![], 123.);
        assert_eq!(op.name(), "Fill");
        assert_eq!(op.input_size(), 0);
        let input_refs = [];
        let expected = Array::fill_f32(&hw, make_shape![], 123.);
        let observed = op.perform(&input_refs).unwrap();
        assert_eq!(observed.shape(), expected.shape());
        assert_eq!(observed.get_scalar_f32(), expected.get_scalar_f32());
    }

    #[test]
    fn test_op_0() {
        let hw = RefCell::new(CpuHardware::new());
        let op = Fill::new(&hw, make_shape![0], 123.);
        assert_eq!(op.name(), "Fill");
        assert_eq!(op.input_size(), 0);
        let input_refs = [];
        let expected = Array::fill_f32(&hw, make_shape![0], 123.);
        let observed = op.perform(&input_refs).unwrap();
        assert_eq!(observed.shape(), expected.shape());
        assert_eq!(observed.get_values_f32(), expected.get_values_f32());
    }

    #[test]
    fn test_op_n() {
        let hw = RefCell::new(CpuHardware::new());
        let op = Fill::new(&hw, make_shape![3], 123.);
        assert_eq!(op.name(), "Fill");
        assert_eq!(op.input_size(), 0);
        let input_refs = [];
        let expected = Array::fill_f32(&hw, make_shape![3], 123.);
        let observed = op.perform(&input_refs).unwrap();
        assert_eq!(observed.shape(), expected.shape());
        assert_eq!(observed.get_values_f32(), expected.get_values_f32());
    }
}

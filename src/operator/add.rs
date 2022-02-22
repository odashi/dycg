use crate::operator::*;

pub(crate) struct Add;

impl Add {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl<'hw> Operator<'hw> for Add {
    fn name(&self) -> String {
        String::from("Add")
    }

    fn input_size(&self) -> usize {
        2
    }

    fn perform_shape(&self, inputs: &[&Shape]) -> Result<Shape> {
        inputs[0].elementwise(inputs[1])
    }

    fn perform(&self, inputs: &[&Array<'hw>]) -> Result<Array<'hw>> {
        inputs[0].elementwise_add_f32(inputs[1])
    }

    fn gradient<'op: 'g, 'g>(
        &self,
        _x: &[Node<'hw, 'op, 'g>],
        _y: Node<'hw, 'op, 'g>,
        gy: Node<'hw, 'op, 'g>,
    ) -> Result<Vec<Node<'hw, 'op, 'g>>>
    where
        'hw: 'op,
    {
        Ok(vec![gy, gy])
    }
}

#[cfg(test)]
mod tests {
    use crate::hardware::cpu::CpuHardware;
    use crate::operator::add::*;

    #[test]
    fn test_properties() {
        let op = Add::new();
        assert_eq!(op.name(), "Add");
        assert_eq!(op.input_size(), 2);
    }

    #[test]
    fn test_perform() {
        let hw = RefCell::new(CpuHardware::new());
        let op = Add::new();
        let inputs = vec![Array::scalar_f32(&hw, 1.), Array::scalar_f32(&hw, 2.)];
        let expected = Array::scalar_f32(&hw, 3.);
        let observed = op.perform(&inputs.iter().collect::<Vec<_>>()).unwrap();
        assert_eq!(observed.shape(), expected.shape());
        assert_eq!(observed.get_scalar_f32(), expected.get_scalar_f32());
    }
}

use crate::array::Array;
use crate::error::Error;
use crate::hardware::Hardware;
use crate::node::Node;
use crate::result::Result;
use crate::shape::Shape;
use std::cell::RefCell;
use std::ptr;

/// Operator represents an individual computation process in the computation graph.
pub(crate) trait Operator<'hw> {
    fn name(&self) -> String;
    fn input_size(&self) -> usize;
    fn perform_shape(&self, inputs: &[&Shape]) -> Result<Shape>;

    fn perform_hardware(
        &self,
        inputs: &[&'hw RefCell<dyn Hardware>],
    ) -> Result<&'hw RefCell<dyn Hardware>> {
        // Most operations assume that all inputs are on the same hardware.
        if self.input_size() > 0 {
            let hw = inputs[0];
            if inputs.iter().skip(1).all(|&x| ptr::eq(hw, x)) {
                Ok(hw)
            } else {
                Err(Error::InvalidNode(format!(
                    "{} does not support operations between different hardwares.",
                    self.name()
                )))
            }
        } else {
            Err(Error::NotSupported(format!(
                "No hardware propagation for {}",
                self.name()
            )))
        }
    }

    fn perform(&self, inputs: &[&Array<'hw>]) -> Result<Array<'hw>>;

    fn gradient<'op: 'g, 'g>(
        &self,
        _x: &[Node<'hw, 'op, 'g>],
        _y: Node<'hw, 'op, 'g>,
        _gy: Node<'hw, 'op, 'g>,
    ) -> Result<Vec<Node<'hw, 'op, 'g>>>
    where
        'hw: 'op,
    {
        Err(Error::NotSupported(format!(
            "No gradient definition for {}",
            self.name(),
        )))
    }
}

// Nullary operators
pub(crate) mod constant;
pub(crate) mod fill;

// Unary operators
pub(crate) mod neg;

// Binary operators
pub(crate) mod add;
pub(crate) mod div;
pub(crate) mod mul;
pub(crate) mod sub;

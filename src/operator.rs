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
    /// Returns the name of the operator.
    ///
    /// # Returns
    ///
    /// The name of the operator.
    fn name(&self) -> String;

    /// Returns the number of required inputs.
    ///
    /// # Returns
    ///
    /// The number of required inputs.
    fn input_size(&self) -> usize;

    /// Calculates the output shape.
    /// This function may be called before `perform()` to propagate `Node` information.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Input shapes. The number of elements must be the same as the return value of
    ///   `input_size()`.
    ///
    /// # Returns:
    ///
    /// * `Ok(Shape)` - The output shape.
    /// * `Err(Error)` - Some error occurred during the process.
    fn perform_shape(&self, inputs: &[&Shape]) -> Result<Shape>;

    /// Calculates the output hardware.
    /// This function may be called before `perform()` to propagate `Node` information.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Input hardwares. The number of elements must be the same as the return value
    ///   of `input_size()`.
    ///
    /// # Returns:
    ///
    /// * `Ok(&RefCell<dyn Hardware>)` - The output hardware.
    /// * `Err(Error)` - Some error occurred during the process.
    fn perform_hardware(
        &self,
        inputs: &[&'hw RefCell<dyn Hardware>],
    ) -> Result<&'hw RefCell<dyn Hardware>> {
        // Most operations assumes that all inputs are on the same hardware.
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

    /// Calculates the output array.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Input arrays. The number of elements must be the same as the return value of
    ///   `input_size()`.
    ///
    /// # Returns:
    ///
    /// * `Ok(Array)` - The ouptut array.
    /// * `Err(Error)` - Some error occurred during the process.
    fn perform(&self, inputs: &[&Array<'hw>]) -> Result<Array<'hw>>;

    /// Constructs the gradient graph.
    ///
    /// # Arguments:
    ///
    /// * `x` - `Node`s of input values. The number of elements must be the same as the return
    ///   value of `input_size()`.
    /// * `y` - `Node` of the output value.
    /// * `gy` - `Node` of the gradient for `y`: df/dy. The target value "f" depends on the
    ///   context when this function is called.
    ///
    /// # Returns:
    ///
    /// List of `Node`s of the gradient for `x[i]`: df/dx[i]. The number of elements must be the
    /// same as the return value of `input_size()`.
    /// Resulting nodes usually represent gy * dy/dx[i] according to the chain rule of derivatives,
    /// but operators can implement other calculation instead for representing unusual gradient
    /// manipulations.
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

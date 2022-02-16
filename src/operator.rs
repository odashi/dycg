use crate::array::Array;
use crate::error::Error;
use crate::node::Node;
use crate::result::Result;

/// Operator represents an individual computation process in the computation graph.
pub(crate) trait Operator<'hw> {
    fn name(&self) -> String;
    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;
    fn perform(&self, inputs: &[&Array<'hw>]) -> Result<Vec<Array<'hw>>>;
    fn gradient<'op, 'g>(
        &self,
        _x: &[&Node<'hw, 'op, 'g>],
        _y: &[&Node<'hw, 'op, 'g>],
        _gy: &[&Node<'hw, 'op, 'g>],
    ) -> Result<Vec<Node<'hw, 'op, 'g>>> {
        Err(Error::NotSupported(format!(
            "No gradient definition for {}",
            self.name(),
        )))
    }
}

pub(crate) mod constant;

pub(crate) mod neg;

pub(crate) mod add;
pub(crate) mod mul;
pub(crate) mod sub;
pub(crate) mod div;
use crate::array::Array;
use crate::error::Error;
use crate::hardware::Hardware;
use crate::operator::{self, Operator};
use crate::result::Result;
use std::cell::RefCell;
use std::fmt;
use std::ptr;

/// Address pointing to an individual value in a Graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeAddress {
    /// Step ID.
    step_id: usize,

    /// Output ID.
    output_id: usize,
}

impl fmt::Display for NodeAddress {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "<{}:{}>", self.step_id, self.output_id)
    }
}

/// Node in an computation graph.
#[derive(Clone, Copy)]
pub struct Node<'hw: 'op, 'op: 'g, 'g> {
    /// Reference to the associated graph.
    graph: &'g RefCell<Graph<'hw, 'op>>,

    /// Address of the value in the graph.
    address: NodeAddress,
}

impl<'hw: 'op, 'op: 'g, 'g> Node<'hw, 'op, 'g> {
    fn new(graph: &'g RefCell<Graph<'hw, 'op>>, address: NodeAddress) -> Self {
        Self { graph, address }
    }

    pub fn from_scalar(
        hardware: &'hw RefCell<dyn Hardware>,
        graph: &'g RefCell<Graph<'hw, 'op>>,
        value: f32,
    ) -> Self {
        Self {
            graph,
            address: graph
                .borrow_mut()
                .add_step(
                    Box::new(operator::Constant::new(Array::new_scalar(hardware, value))),
                    vec![],
                )
                .unwrap()
                .pop()
                .unwrap(),
        }
    }

    pub fn to_scalar(&self) -> f32 {
        self.graph
            .borrow_mut()
            .calculate(&self.address)
            .unwrap()
            .to_scalar()
            .unwrap()
    }

    pub fn check_graph(&self, others: &[&Self]) -> Result<&'g RefCell<Graph<'hw, 'op>>> {
        others
            .iter()
            .all(|&o| ptr::eq(self.graph, o.graph))
            .then(|| self.graph)
            .ok_or(Error::InvalidGraph(format!(
                "Attempted calculation between Nodes on different Graph."
            )))
    }
}

impl<'hw: 'op, 'op: 'g, 'g> fmt::Display for Node<'hw, 'op, 'g> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:016p}:{}", self.graph, self.address)
    }
}

impl<'hw: 'op, 'op: 'g, 'g> fmt::Debug for Node<'hw, 'op, 'g> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:016p}:{}", self.graph, self.address)
    }
}

impl<'hw: 'op, 'op: 'g, 'g> PartialEq for Node<'hw, 'op, 'g> {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self.graph, other.graph) && self.address == other.address
    }
}

impl<'hw: 'op, 'op: 'g, 'g> Eq for Node<'hw, 'op, 'g> {}

//// Individual step in computation graphs.
/// Step owns an Operator which consumes several values produced by preceding steps.
pub(crate) struct Step<'hw: 'op, 'op> {
    /// Operator owned by this step.
    operator: Box<dyn Operator<'hw> + 'op>,

    /// Input nodes.
    inputs: Vec<NodeAddress>,

    /// Output values.
    outputs: Option<Vec<Array<'hw>>>,
}

impl<'hw: 'op, 'op> Step<'hw, 'op> {
    /// Creates a new `Step` object.
    fn new(
        operator: Box<dyn Operator<'hw> + 'op>,
        inputs: Vec<NodeAddress>,
        outputs: Option<Vec<Array<'hw>>>,
    ) -> Self {
        Self {
            operator,
            inputs,
            outputs,
        }
    }
}

// Computation graph.
pub struct Graph<'hw: 'op, 'op> {
    /// All steps registered to this graph.
    steps: Vec<Step<'hw, 'op>>,
}

impl<'hw: 'op, 'op> Graph<'hw, 'op> {
    /// Creates a new empty `Graph` object.
    ///
    /// # Returns
    ///
    /// A new `Graph` object containing zero steps.
    fn new() -> Self {
        Self { steps: vec![] }
    }

    /// Checks if the `NodeAddress` is valid for this Graph.
    ///
    /// # Arguments
    ///
    /// * `address` - `NodeAddress` object to be checked.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - The `address` is valid in this graph.
    /// * `Err(Error)` - The `address` is invalid for returned reason.
    fn check_address(&self, addr: &NodeAddress) -> Result<()> {
        if addr.step_id >= self.steps.len() {
            return Err(Error::InvalidNode(format!(
                "NodeAddress {} does not point to a valid node.",
                addr
            )));
        }
        unsafe {
            if addr.output_id
                >= self
                    .steps
                    .get_unchecked(addr.step_id)
                    .operator
                    .output_size()
            {
                return Err(Error::InvalidNode(format!(
                    "NodeAddress {} points to a valid node, but does not point to a valid output.",
                    addr
                )));
            }
        }
        Ok(())
    }

    /// Inserts a new step into this graph.
    ///
    /// # Arguments
    ///
    /// * `operator` - `Operator` for the new step.
    /// * `inputs` - Input nodes.
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<Node>)` - A new step is inserted correctly. Each `Node` in the returned value
    ///   points to output values.
    /// * `Err(Error)` - Some error occurred during the process.
    pub(crate) fn add_step(
        &mut self,
        operator: Box<dyn Operator<'hw> + 'op>,
        inputs: Vec<NodeAddress>,
    ) -> Result<Vec<NodeAddress>> {
        let new_step_id = self.steps.len();
        let input_size = operator.input_size();
        let output_size = operator.output_size();

        if inputs.len() != input_size {
            return Err(Error::InvalidLength(format!(
                "Operator requires {} inputs, but got {}.",
                operator.input_size(),
                inputs.len()
            )));
        }

        if output_size == 0 {
            return Err(Error::OutOfRange(format!(
                "Operator must return at least 1 output."
            )));
        }

        for &address in &inputs {
            self.check_address(&address)?;
        }

        self.steps.push(Step::new(operator, inputs, None));

        Ok((0..output_size)
            .map(|output_id| NodeAddress {
                step_id: new_step_id,
                output_id,
            })
            .collect())
    }

    /// Checks if the specified step is already calculated or not.
    ///
    /// # Arguments
    ///
    /// * `step_id` - ID of the step to check.
    ///
    /// # Returns
    ///
    /// * `true` - The step is already calculated.
    /// * `false` - Otherwise.
    ///
    /// # Requirements
    ///
    /// The specified step ID is valid in the graph.
    unsafe fn is_calculated_unchecked(&self, step_id: usize) -> bool {
        self.steps.get_unchecked(step_id).outputs.is_some()
    }

    /// Obtains associated value of the node.
    ///
    /// # Arguments
    ///
    /// * `address` - Target node to obtain the calculated value.
    ///
    /// # Returns
    ///
    /// Reference to the calculated value.
    ///
    /// # Requirements
    ///
    /// * The specified `address` (step/output IDs) is valid in the graph.
    /// * The associated step is already calculated.
    unsafe fn get_value_unchecked(&self, address: NodeAddress) -> Array<'hw> {
        self.steps
            .get_unchecked(address.step_id)
            .outputs
            .as_ref()
            .unwrap()
            .get_unchecked(address.output_id)
            .clone()
    }

    /// Performs calculation to obtain the value of specified node.
    ///
    /// This function internally performs a push-down automaton to recursively obtain the values
    /// necessary to calculate the target node.
    /// If nodes (both target/parents) are already calculated, calculation is skipped and the cached
    /// values are used instead.
    ///
    /// # Arguments
    ///
    /// * `target` - Target address to obtain the value.
    ///
    /// # Returns
    ///
    /// * Calculated/cached value associated to `target`.
    pub(crate) fn calculate(&mut self, target: &NodeAddress) -> Result<Array<'hw>> {
        self.check_address(target)?;

        // Actions for the push-down automaton representing the following procedure:
        /*
            fn calculate(node) {
                if is_cached(node) { return; }
                for input in node.inputs { calculate(input); }
                perform(node);
            }
        */
        enum Action {
            Fetch,
            Perform,
        }

        // action_stack represents the state of the push-down automaton.
        let mut action_stack = vec![(target.step_id, Action::Fetch)];

        while !action_stack.is_empty() {
            let (step_id, action) = action_stack.pop().unwrap();
            match action {
                Action::Fetch => {
                    unsafe {
                        // If the node holds a value already, we need to do nothing.
                        if self.is_calculated_unchecked(step_id) {
                            continue;
                        }

                        // We need to perform the operator.
                        action_stack.push((step_id, Action::Perform));

                        // Before performing the operator, we need all inputs to be calculated.
                        for &input_node in &self.steps.get_unchecked(step_id).inputs {
                            action_stack.push((input_node.step_id, Action::Fetch));
                        }
                    }
                }
                Action::Perform => {
                    unsafe {
                        let node = self.steps.get_unchecked(step_id);

                        // Collect the input values.
                        // At this point, all the inputs should be calculated already.
                        let inputs = node
                            .inputs
                            .iter()
                            .map(|node| {
                                &self.steps[node.step_id].outputs.as_ref().unwrap()[node.output_id]
                            })
                            .collect::<Vec<_>>();

                        // Perform the operator.
                        self.steps.get_unchecked_mut(step_id).outputs =
                            Some(node.operator.perform(&inputs).unwrap());
                    }
                }
            }
        }

        Ok(unsafe { self.get_value_unchecked(*target) })
    }
}

/// "+" operator for `Node`.
impl<'hw: 'op, 'op: 'g, 'g> std::ops::Add for Node<'hw, 'op, 'g> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            graph: self.graph,
            address: self
                .check_graph(&[&other])
                .unwrap()
                .borrow_mut()
                .add_step(
                    Box::new(operator::Add::new()),
                    vec![self.address, other.address],
                )
                .unwrap()[0],
        }
    }
}

/// "-" operator for `Node`.
impl<'hw: 'op, 'op: 'g, 'g> std::ops::Sub for Node<'hw, 'op, 'g> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            graph: self.graph,
            address: self
                .check_graph(&[&other])
                .unwrap()
                .borrow_mut()
                .add_step(
                    Box::new(operator::Sub::new()),
                    vec![self.address, other.address],
                )
                .unwrap()[0],
        }
    }
}

/// "*" operator for `Node`.
impl<'hw: 'op, 'op: 'g, 'g> std::ops::Mul for Node<'hw, 'op, 'g> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            graph: self.graph,
            address: self
                .check_graph(&[&other])
                .unwrap()
                .borrow_mut()
                .add_step(
                    Box::new(operator::Mul::new()),
                    vec![self.address, other.address],
                )
                .unwrap()[0],
        }
    }
}

/// "/" operator for `Node`.
impl<'hw: 'op, 'op: 'g, 'g> std::ops::Div for Node<'hw, 'op, 'g> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self {
            graph: self.graph,
            address: self
                .check_graph(&[&other])
                .unwrap()
                .borrow_mut()
                .add_step(
                    Box::new(operator::Div::new()),
                    vec![self.address, other.address],
                )
                .unwrap()[0],
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::*;
    use crate::hardware::cpu::CpuHardware;
    use crate::make_shape;
    use std::cell::RefCell;

    #[test]
    fn test_steps() {
        let hw = RefCell::new(CpuHardware::new("test"));
        let g = RefCell::new(Graph::new());
        let lhs = Node::from_scalar(&hw, &g, 1.);
        let rhs = Node::from_scalar(&hw, &g, 2.);
        let ret = lhs + rhs;

        #[rustfmt::skip]
        (|| {
            assert_eq!(lhs, Node::new(&g, NodeAddress { step_id: 0, output_id: 0 }));
            assert_eq!(rhs, Node::new(&g, NodeAddress { step_id: 1, output_id: 0 }));
            assert_eq!(ret, Node::new(&g, NodeAddress { step_id: 2, output_id: 0 }));
        })();

        assert_eq!(g.borrow().steps.len(), 3);
        let retval = g.borrow_mut().calculate(&ret.address).unwrap();
        assert_eq!(*retval.shape(), make_shape![]);
        assert_eq!(retval.to_scalar(), Ok(3.));
    }

    #[test]
    fn test_add() {
        let hw = RefCell::new(CpuHardware::new("test"));
        let g = RefCell::new(Graph::new());

        let lhs = Node::from_scalar(&hw, &g, 1.);
        let rhs = Node::from_scalar(&hw, &g, 2.);
        let ret = lhs + rhs;
        assert_eq!(
            g.borrow_mut().calculate(&ret.address).unwrap().to_scalar(),
            Ok(3.)
        );
    }

    #[test]
    fn test_sub() {
        let hw = RefCell::new(CpuHardware::new("test"));
        let g = RefCell::new(Graph::new());

        let lhs = Node::from_scalar(&hw, &g, 1.);
        let rhs = Node::from_scalar(&hw, &g, 2.);
        let ret = lhs - rhs;
        assert_eq!(
            g.borrow_mut().calculate(&ret.address).unwrap().to_scalar(),
            Ok(-1.)
        );
    }

    #[test]
    fn test_mul() {
        let hw = RefCell::new(CpuHardware::new("test"));
        let g = RefCell::new(Graph::new());

        let lhs = Node::from_scalar(&hw, &g, 1.);
        let rhs = Node::from_scalar(&hw, &g, 2.);
        let ret = lhs * rhs;
        assert_eq!(
            g.borrow_mut().calculate(&ret.address).unwrap().to_scalar(),
            Ok(2.)
        );
    }

    #[test]
    fn test_div() {
        let hw = RefCell::new(CpuHardware::new("test"));
        let g = RefCell::new(Graph::new());

        let lhs = Node::from_scalar(&hw, &g, 1.);
        let rhs = Node::from_scalar(&hw, &g, 2.);
        let ret = lhs / rhs;
        assert_eq!(
            g.borrow_mut().calculate(&ret.address).unwrap().to_scalar(),
            Ok(0.5)
        );
    }

    #[test]
    fn test_multiple_computation() {
        let hw = RefCell::new(CpuHardware::new("test"));
        let g = RefCell::new(Graph::new());

        let a = Node::from_scalar(&hw, &g, 1.);
        let b = Node::from_scalar(&hw, &g, 2.);
        let c = Node::from_scalar(&hw, &g, 3.);
        let y = a + b * c;
        assert_eq!(
            g.borrow_mut().calculate(&y.address).unwrap().to_scalar(),
            Ok(7.)
        );
    }
}

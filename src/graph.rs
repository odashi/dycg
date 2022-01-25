use std::fmt;
use std::rc::Rc;

use scoped_tls_hkt::scoped_thread_local;

use crate::array::{make_cpu_scalar, Array};
use crate::error::Error;
use crate::operator::{self, Operator};
use crate::result::Result;

scoped_thread_local!(static mut CURRENT_GRAPH: Graph);

/// Individual step in computation graphs.
/// Step owns an Operator which consumes several values produced by preceding steps.
pub(crate) struct Step {
    /// Operator owned by this step.
    operator: Box<dyn Operator>,

    /// Input nodes.
    inputs: Vec<Node>,

    /// Output values.
    outputs: Option<Vec<Rc<Array>>>,
}

/// Node in an computation graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Node {
    step_id: usize,
    output_id: usize,
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "<{}:{}>", self.step_id, self.output_id)
    }
}

/// Computation graph.
pub(crate) struct Graph {
    /// All steps registered to this graph.
    steps: Vec<Step>,
}

impl Graph {
    /// Creates a new empty `Graph` object.
    ///
    /// # Returns
    ///
    /// A new `Graph` object containing zero steps.
    fn new() -> Self {
        Self { steps: vec![] }
    }

    /// Checks if the `Node` is valid for this Graph.
    ///
    /// # Arguments
    ///
    /// * `node` - `Node` object to be checked.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - The `node` is valid in this graph.
    /// * `Err(Error)` - The `node` is invalid for returned reason.
    fn check_node(&self, node: Node) -> Result<()> {
        if node.step_id >= self.steps.len() {
            return Err(Error::InvalidNode(format!(
                "Node {} does not point to a valid node.",
                node
            )));
        }
        unsafe {
            if node.output_id
                >= self
                    .steps
                    .get_unchecked(node.step_id)
                    .operator
                    .output_size()
            {
                return Err(Error::InvalidNode(format!(
                    "Node {} points to a valid node, but does not point to a valid output.",
                    node
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
        operator: Box<dyn Operator>,
        inputs: Vec<Node>,
    ) -> Result<Vec<Node>> {
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

        for &node in &inputs {
            self.check_node(node)?;
        }

        self.steps.push(Step {
            operator,
            inputs,
            outputs: None,
        });

        Ok((0..output_size)
            .map(|output_id| Node {
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
    /// * `node` - Target node to obtain the calculated value.
    ///
    /// # Returns
    ///
    /// Reference to the calculated value.
    ///
    /// # Requirements
    ///
    /// * The specified node (step/output IDs) is valid in the graph.
    /// * The associated step is already calculated.
    unsafe fn get_value_unchecked(&self, node: Node) -> &Rc<Array> {
        self.steps
            .get_unchecked(node.step_id)
            .outputs
            .as_ref()
            .unwrap()
            .get_unchecked(node.output_id)
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
    /// * `target` - Target node to obtain the value.
    ///
    /// # Returns
    ///
    /// * Calculated/cached value associated to `target`.
    pub(crate) fn calculate(&mut self, target: Node) -> Result<&Rc<Array>> {
        self.check_node(target)?;

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
                        // If the node holds a value already, we don't need to do nothing.
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
                                self.steps
                                    .get_unchecked(node.step_id)
                                    .outputs
                                    .as_ref()
                                    .unwrap()
                                    .get_unchecked(node.output_id)
                                    .as_ref()
                            })
                            .collect::<Vec<&Array>>();

                        // Perform the operator.
                        self.steps.get_unchecked_mut(step_id).outputs =
                            Some(node.operator.perform(&inputs).unwrap());
                    }
                }
            }
        }

        Ok(unsafe { self.get_value_unchecked(target) })
    }
}

/// Enters a new runtime context with a new `Graph` object.
///
/// This function manages a lifetime of the global context of `Graph`.
/// A `Graph` is only available within the provided closure to this function.
///
/// By the API design this function can be called recursively, but the actual behavior is not
/// supported.
///
/// For concurrency, this function supports multithreading, but the actual behavior is not
/// guaranteed. This function does not also support asynchronous calls for now.
///
/// # Arguments
///
/// * `callback` - Callback procedure for the new runtime context. The same `Graph` object is
///   associated during the `callback` is processed unless this function is called recursively.
pub fn trace(callback: impl FnOnce() -> ()) -> () {
    CURRENT_GRAPH.set(&mut Graph::new(), callback);
}

/// Enters a new runtime context by obtaining the current `Graph` object.
///
/// This function can not be called recursively with only itself.
///
/// During processing the associated callback, this function releases the current `Graph` from the
/// context, and it will be restored at finalizing this function.
///
/// # Arguments
///
/// * `callback` - Callback procedure for the new runtime context. `callback` receives the current
///   `Graph` object and it can be used during its scope.
///
/// # Returns
///
/// Returned value from the `callback`.
pub(crate) fn with_current_graph<T>(callback: impl FnOnce(&mut Graph) -> T) -> T {
    CURRENT_GRAPH.with(callback)
}

/// Direct conversion from a floating nubmer to `Node`.
impl From<f32> for Node {
    fn from(src: f32) -> Self {
        with_current_graph(|g| unsafe {
            *g.add_step(
                Box::new(operator::Constant::new(make_cpu_scalar(src))),
                vec![],
            )
            .unwrap()
            .get_unchecked(0)
        })
    }
}

/// Reverse conversion from `Node` to a floating number.
impl From<Node> for f32 {
    fn from(src: Node) -> Self {
        with_current_graph(|g| g.calculate(src).unwrap().into_scalar().unwrap())
    }
}

/// "+" operator for `Node`.
impl std::ops::Add for Node {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        with_current_graph(|g| unsafe {
            *g.add_step(Box::new(operator::Add::new()), vec![self, other])
                .unwrap()
                .get_unchecked(0)
        })
    }
}

/// "-" operator for `Node`.
impl std::ops::Sub for Node {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        with_current_graph(|g| unsafe {
            *g.add_step(Box::new(operator::Sub::new()), vec![self, other])
                .unwrap()
                .get_unchecked(0)
        })
    }
}

/// "*" operator for `Node`.
impl std::ops::Mul for Node {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        with_current_graph(|g| unsafe {
            *g.add_step(Box::new(operator::Mul::new()), vec![self, other])
                .unwrap()
                .get_unchecked(0)
        })
    }
}

/// "/" operator for `Node`.
impl std::ops::Div for Node {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        with_current_graph(|g| unsafe {
            *g.add_step(Box::new(operator::Div::new()), vec![self, other])
                .unwrap()
                .get_unchecked(0)
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::*;
    use crate::make_shape;

    #[test]
    fn test_steps() {
        trace(|| {
            let lhs = Node::from(1.);
            let rhs = Node::from(2.);
            let ret = lhs + rhs;

            #[rustfmt::skip]
            (|| {
                assert_eq!(lhs, Node { step_id: 0, output_id: 0 });
                assert_eq!(rhs, Node { step_id: 1, output_id: 0 });
                assert_eq!(ret, Node { step_id: 2, output_id: 0 });
            })();

            with_current_graph(|g| {
                assert_eq!(g.steps.len(), 3);
                let retval = g.calculate(ret).unwrap();
                assert_eq!(*retval.shape(), make_shape![]);
                assert_eq!(retval.into_scalar(), Ok(3.));
            });
        });
    }

    #[test]
    fn test_add() {
        trace(|| {
            let lhs = Node::from(1.);
            let rhs = Node::from(2.);
            let ret = lhs + rhs;
            assert_eq!(f32::from(ret), 3.);
        });
    }

    #[test]
    fn test_sub() {
        trace(|| {
            let lhs = Node::from(1.);
            let rhs = Node::from(2.);
            let ret = lhs - rhs;
            assert_eq!(f32::from(ret), -1.);
        });
    }

    #[test]
    fn test_mul() {
        trace(|| {
            let lhs = Node::from(1.);
            let rhs = Node::from(2.);
            let ret = lhs * rhs;
            assert_eq!(f32::from(ret), 2.);
        });
    }

    #[test]
    fn test_div() {
        trace(|| {
            let lhs = Node::from(1.);
            let rhs = Node::from(2.);
            let ret = lhs / rhs;
            assert_eq!(f32::from(ret), 0.5);
        });
    }

    #[test]
    fn test_multiple_computation() {
        trace(|| {
            let a = Node::from(1.);
            let b = Node::from(2.);
            let c = Node::from(3.);
            let y = a + b * c;
            assert_eq!(f32::from(y), 7.);
        });
    }
}

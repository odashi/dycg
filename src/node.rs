use crate::array::Array;
use crate::error::Error;
use crate::graph::Graph;
use crate::hardware::Hardware;
use crate::operator;
use crate::result::Result;
use crate::shape::Shape;
use std::cell::RefCell;
use std::fmt;
use std::ptr;

/// Node in a computation graph.
#[derive(Clone, Copy)]
pub struct Node<'hw: 'op, 'op: 'g, 'g> {
    /// Reference to the associated graph.
    graph: &'g RefCell<Graph<'hw, 'op>>,

    /// step_id of the value in the graph.
    step_id: usize,
}

impl<'hw: 'op, 'op: 'g, 'g> Node<'hw, 'op, 'g> {
    fn new(graph: &'g RefCell<Graph<'hw, 'op>>, step_id: usize) -> Self {
        Self { graph, step_id }
    }

    pub fn from_scalar(
        hardware: &'hw RefCell<dyn Hardware>,
        graph: &'g RefCell<Graph<'hw, 'op>>,
        value: f32,
    ) -> Self {
        Self::new(
            graph,
            graph
                .borrow_mut()
                .add_step(
                    Box::new(operator::constant::Constant::new(Array::scalar_f32(
                        hardware, value,
                    ))),
                    vec![],
                )
                .unwrap(),
        )
    }

    pub fn to_scalar(&self) -> f32 {
        self.graph
            .borrow_mut()
            .calculate(self.step_id)
            .unwrap()
            .get_scalar_f32()
            .unwrap()
    }

    pub fn check_graph(&self, others: &[&Self]) -> Result<&'g RefCell<Graph<'hw, 'op>>> {
        others
            .iter()
            .all(|&o| ptr::eq(self.graph, o.graph))
            .then(|| self.graph)
            .ok_or_else(|| {
                Error::InvalidGraph(
                    "Attempted calculation between Nodes on different Graph.".to_string(),
                )
            })
    }

    pub fn shape(&self) -> Shape {
        self.graph
            .borrow()
            .get_step(self.step_id)
            .unwrap()
            .output
            .shape()
            .clone()
    }

    pub fn hardware(&self) -> &'hw RefCell<dyn Hardware> {
        self.graph
            .borrow()
            .get_step(self.step_id)
            .unwrap()
            .output
            .hardware()
    }

    pub fn calculate(&self) -> Result<Array<'hw>> {
        self.graph.borrow_mut().calculate(self.step_id)
    }

    /// Registers `Fill` operation to the graph.
    ///
    /// # Arguments
    ///
    /// * `hardware` - `Hardware` object to hold the value.
    /// * `graph` - `Graph` object to register the operation.
    /// * `shape` - `Shape` of the output array.
    /// * `value` - Value of each element in the output array.
    pub fn fill(
        hardware: &'hw RefCell<dyn Hardware>,
        graph: &'g RefCell<Graph<'hw, 'op>>,
        shape: Shape,
        value: f32,
    ) -> Self {
        Self::new(
            graph,
            graph
                .borrow_mut()
                .add_step(
                    Box::new(operator::fill::Fill::new(hardware, shape, value)),
                    vec![],
                )
                .unwrap(),
        )
    }
}

impl<'hw: 'op, 'op: 'g, 'g> fmt::Display for Node<'hw, 'op, 'g> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:016p}:{}", self.graph, self.step_id)
    }
}

impl<'hw: 'op, 'op: 'g, 'g> fmt::Debug for Node<'hw, 'op, 'g> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:016p}:{}", self.graph, self.step_id)
    }
}

impl<'hw: 'op, 'op: 'g, 'g> PartialEq for Node<'hw, 'op, 'g> {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self.graph, other.graph) && self.step_id == other.step_id
    }
}

impl<'hw: 'op, 'op: 'g, 'g> Eq for Node<'hw, 'op, 'g> {}

/// Unary "-" operator for `Node`.
impl<'hw: 'op, 'op: 'g, 'g> std::ops::Neg for Node<'hw, 'op, 'g> {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            graph: self.graph,
            step_id: self
                .check_graph(&[])
                .unwrap()
                .borrow_mut()
                .add_step(Box::new(operator::neg::Neg::new()), vec![self.step_id])
                .unwrap(),
        }
    }
}

/// "+" operator for `Node`.
impl<'hw: 'op, 'op: 'g, 'g> std::ops::Add for Node<'hw, 'op, 'g> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            graph: self.graph,
            step_id: self
                .check_graph(&[&other])
                .unwrap()
                .borrow_mut()
                .add_step(
                    Box::new(operator::add::Add::new()),
                    vec![self.step_id, other.step_id],
                )
                .unwrap(),
        }
    }
}

/// "-" operator for `Node`.
impl<'hw: 'op, 'op: 'g, 'g> std::ops::Sub for Node<'hw, 'op, 'g> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            graph: self.graph,
            step_id: self
                .check_graph(&[&other])
                .unwrap()
                .borrow_mut()
                .add_step(
                    Box::new(operator::sub::Sub::new()),
                    vec![self.step_id, other.step_id],
                )
                .unwrap(),
        }
    }
}

/// "*" operator for `Node`.
impl<'hw: 'op, 'op: 'g, 'g> std::ops::Mul for Node<'hw, 'op, 'g> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            graph: self.graph,
            step_id: self
                .check_graph(&[&other])
                .unwrap()
                .borrow_mut()
                .add_step(
                    Box::new(operator::mul::Mul::new()),
                    vec![self.step_id, other.step_id],
                )
                .unwrap(),
        }
    }
}

/// "/" operator for `Node`.
impl<'hw: 'op, 'op: 'g, 'g> std::ops::Div for Node<'hw, 'op, 'g> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self {
            graph: self.graph,
            step_id: self
                .check_graph(&[&other])
                .unwrap()
                .borrow_mut()
                .add_step(
                    Box::new(operator::div::Div::new()),
                    vec![self.step_id, other.step_id],
                )
                .unwrap(),
        }
    }
}

/// Calculates the value of the derivative dy/dx.
///
/// # Arguments
///
/// * `y` - `Node` representing the output value.
/// * `x` - List of `Node`s representing the input value.
///
/// # Returns
///
/// New `Node`s representing the derivative dy/dx. The order of elements corresponds to that of
/// `x`.
///
/// # Panics
///
/// * Attempting to calculate gradients between nodes on different graphs.
/// * Some nodes hold invalid information.
pub fn grad<'hw, 'op, 'g>(
    y: Node<'hw, 'op, 'g>,
    x: &[Node<'hw, 'op, 'g>],
) -> Vec<Node<'hw, 'op, 'g>> {
    // Strategy: calculates gradients of every step between the earliest step in `x` and `y`.
    // This is redundant because some steps may not belong to the path between any of `x` and `y`,
    // But it may be enough efficient because the usual use-case of this function may be
    // "calculating graditns from the last step to every input."

    let g = y.graph;
    assert!(
        x.iter().all(|node| ptr::eq(node.graph, g)),
        "Gradients can not be calculated beyond different graphs."
    );

    let first_step_id = match x.iter().map(|node| node.step_id).min() {
        Some(step_id) => step_id,
        None => return vec![], // `x` is empty. No need to calculate any gradients.
    };
    let last_step_id = y.step_id;

    // Placeholder of gradient nodes.
    let mut gradients = vec![None; g.borrow().num_steps()];

    // Assigns the gradient of `y` == 1.
    *(unsafe { gradients.get_unchecked_mut(last_step_id) }) =
        Some(Node::fill(y.hardware(), g, y.shape(), 1.));

    // Performs backpropagation.
    for step_id in ((first_step_id + 1)..=last_step_id).rev() {
        let cur_gy = match unsafe { gradients.get_unchecked(step_id) } {
            Some(node) => *node,
            None => continue, // No preceding gradients propagated to this step.
        };

        let (cur_xs_ids, maybe_grad_fn) = {
            let g = g.borrow();
            let step = g.get_step(step_id).unwrap();
            (step.inputs.clone(), step.operator.get_gradient_fn())
        };

        let grad_fn = match maybe_grad_fn {
            Some(f) => f,
            None => continue, // No gradient operation is defined for this step.
        };

        // Calculates gradients for this step.
        let cur_xs = cur_xs_ids
            .iter()
            .map(|&step_id| Node::new(g, step_id))
            .collect::<Vec<_>>();
        let cur_y = Node::new(g, step_id);
        let cur_gxs = grad_fn.perform(&cur_xs, cur_y, cur_gy);

        // Integrates gradients.
        for (&cur_x_id, &cur_gx) in cur_xs_ids.iter().zip(cur_gxs.iter()) {
            let prev_gx = unsafe { gradients.get_unchecked_mut(cur_x_id) };
            *prev_gx = match prev_gx {
                Some(node) => Some(*node + cur_gx),
                None => Some(cur_gx),
            }
        }
    }

    // Collects the nodes representing gradients of `x`.
    x.iter()
        .map(|node| {
            match unsafe { gradients.get_unchecked(node.step_id) } {
                Some(grad_node) => *grad_node,
                // No gradient propagation occurred for this node,
                // assuming that the gradient is 0.
                None => Node::fill(node.hardware(), g, node.shape(), 0.),
            }
        })
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod grad_tests;

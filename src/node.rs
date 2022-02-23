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
/// * `xs` - List of `Node`s representing the input value.
///
/// # Returns
///
/// New `Node`s representing the derivative dy/dx. The order of elements corresponds to that of
/// `x`.
pub fn grad<'hw, 'op, 'g>(
    y: Node<'hw, 'op, 'g>,
    x: &[Node<'hw, 'op, 'g>],
) -> Result<Vec<Node<'hw, 'op, 'g>>> {
    // Strategy: calculates gradients of every step between the earliest step in `x` and `y`.
    // This is redundant because some steps may not belong to the path between any of `x` and `y`,
    // But it may be enough efficient because the usual use-case of this function may be
    // "calculating graditns from the last step to every input."

    let g = y.graph;
    if !x.iter().all(|node| ptr::eq(node.graph, g)) {
        return Err(Error::InvalidNode(
            "Gradients can not be calculated beyond different graphs.".to_string(),
        ));
    }

    let earliest_step_id = if let Some(step_id) = x.iter().map(|node| node.step_id).min() {
        step_id
    } else {
        // `x` is empty. No need to calculate any gradients.
        return Ok(vec![]);
    };
    let latest_step_id = y.step_id;

    // Placeholder of gradient nodes.
    let mut gradients = vec![None; g.borrow().num_steps()];

    // Assigns the gradient of `y` == 1.
    *(unsafe { gradients.get_unchecked_mut(latest_step_id) }) =
        Some(Node::fill(y.hardware(), g, y.shape(), 1.));

    // Performs backpropagation.
    for step_id in ((earliest_step_id + 1)..=latest_step_id).rev() {
        let cur_gy = unsafe { gradients.get_unchecked(step_id) };
        if cur_gy.is_none() {
            // No gradients propagated from the outputs.
            // We can skip this backward step.
            continue;
        }

        let cur_y = Node::new(g, step_id);

        let (cur_xs_ids, grad_fn) = {
            let g = g.borrow();
            let step = g.get_step(step_id).unwrap();
            (step.inputs.clone(), step.operator.get_gradient_fn())
        };

        if grad_fn.is_none() {
            // No gradient operation is defined for this step.
            continue;
        }

        let cur_xs = cur_xs_ids
            .iter()
            .map(|&step_id| Node::new(g, step_id))
            .collect::<Vec<_>>();

        // Obtains nodes for gradients by this step.
        let cur_gxs = grad_fn.unwrap().perform(&cur_xs, cur_y, cur_gy.unwrap());

        // Integrates gradients.
        for (&cur_x_id, &cur_gx) in cur_xs_ids.iter().zip(cur_gxs.iter()) {
            let prev_gx = unsafe { gradients.get_unchecked_mut(cur_x_id) };
            *prev_gx = match prev_gx {
                Some(node) => Some(*node + cur_gx),
                None => Some(cur_gx),
            }
        }
    }

    // Collects the nodes.
    Ok(x.iter()
        .map(|node| {
            match unsafe { gradients.get_unchecked(node.step_id) } {
                Some(grad_node) => *grad_node,
                // If no gradient propagation occurred for this node,
                // we assume that the gradient is 0.
                None => Node::fill(node.hardware(), g, node.shape(), 0.),
            }
        })
        .collect::<Vec<_>>())
}

#[cfg(test)]
mod tests {
    use crate::hardware::cpu::CpuHardware;
    use crate::make_shape;
    use crate::node::*;

    #[test]
    fn test_steps() {
        let hw = RefCell::new(CpuHardware::new());
        let g = RefCell::new(Graph::new());
        let lhs = Node::from_scalar(&hw, &g, 1.);
        let rhs = Node::from_scalar(&hw, &g, 2.);
        let ret = lhs + rhs;

        assert_eq!(lhs, Node::new(&g, 0));
        assert_eq!(rhs, Node::new(&g, 1));
        assert_eq!(ret, Node::new(&g, 2));
        assert_eq!(lhs.shape(), make_shape![]);
        assert_eq!(rhs.shape(), make_shape![]);
        assert_eq!(ret.shape(), make_shape![]);
        assert!(ptr::eq(lhs.hardware(), &hw));
        assert!(ptr::eq(rhs.hardware(), &hw));
        assert!(ptr::eq(ret.hardware(), &hw));

        {
            let g = g.borrow();
            assert_eq!(g.num_steps(), 3);
            assert_eq!(g.get_step(0).unwrap().operator.name(), "Constant");
            assert_eq!(g.get_step(1).unwrap().operator.name(), "Constant");
            assert_eq!(g.get_step(2).unwrap().operator.name(), "Add");
        }

        let retval = ret.calculate().unwrap();
        assert_eq!(*retval.shape(), make_shape![]);
        assert_eq!(retval.get_scalar_f32(), Ok(3.));
    }

    #[test]
    fn test_neg() {
        let hw = RefCell::new(CpuHardware::new());
        let g = RefCell::new(Graph::new());

        let src = Node::from_scalar(&hw, &g, 42.);
        let dest = -src;

        assert_eq!(src.shape(), make_shape![]);
        assert_eq!(dest.shape(), make_shape![]);
        assert!(ptr::eq(src.hardware(), &hw));
        assert!(ptr::eq(dest.hardware(), &hw));

        assert_eq!(dest.calculate().unwrap().get_scalar_f32(), Ok(-42.));
    }

    #[test]
    fn test_add() {
        let hw = RefCell::new(CpuHardware::new());
        let g = RefCell::new(Graph::new());

        let lhs = Node::from_scalar(&hw, &g, 1.);
        let rhs = Node::from_scalar(&hw, &g, 2.);
        let ret = lhs + rhs;

        assert_eq!(lhs.shape(), make_shape![]);
        assert_eq!(rhs.shape(), make_shape![]);
        assert_eq!(ret.shape(), make_shape![]);
        assert!(ptr::eq(lhs.hardware(), &hw));
        assert!(ptr::eq(rhs.hardware(), &hw));
        assert!(ptr::eq(ret.hardware(), &hw));

        assert_eq!(ret.calculate().unwrap().get_scalar_f32(), Ok(3.));
    }

    #[test]
    fn test_sub() {
        let hw = RefCell::new(CpuHardware::new());
        let g = RefCell::new(Graph::new());

        let lhs = Node::from_scalar(&hw, &g, 1.);
        let rhs = Node::from_scalar(&hw, &g, 2.);
        let ret = lhs - rhs;

        assert_eq!(lhs.shape(), make_shape![]);
        assert_eq!(rhs.shape(), make_shape![]);
        assert_eq!(ret.shape(), make_shape![]);
        assert!(ptr::eq(lhs.hardware(), &hw));
        assert!(ptr::eq(rhs.hardware(), &hw));
        assert!(ptr::eq(ret.hardware(), &hw));

        assert_eq!(ret.calculate().unwrap().get_scalar_f32(), Ok(-1.));
    }

    #[test]
    fn test_mul() {
        let hw = RefCell::new(CpuHardware::new());
        let g = RefCell::new(Graph::new());

        let lhs = Node::from_scalar(&hw, &g, 1.);
        let rhs = Node::from_scalar(&hw, &g, 2.);
        let ret = lhs * rhs;

        assert_eq!(lhs.shape(), make_shape![]);
        assert_eq!(rhs.shape(), make_shape![]);
        assert_eq!(ret.shape(), make_shape![]);
        assert!(ptr::eq(lhs.hardware(), &hw));
        assert!(ptr::eq(rhs.hardware(), &hw));
        assert!(ptr::eq(ret.hardware(), &hw));

        assert_eq!(ret.calculate().unwrap().get_scalar_f32(), Ok(2.));
    }

    #[test]
    fn test_div() {
        let hw = RefCell::new(CpuHardware::new());
        let g = RefCell::new(Graph::new());

        let lhs = Node::from_scalar(&hw, &g, 1.);
        let rhs = Node::from_scalar(&hw, &g, 2.);
        let ret = lhs / rhs;

        assert_eq!(lhs.shape(), make_shape![]);
        assert_eq!(rhs.shape(), make_shape![]);
        assert_eq!(ret.shape(), make_shape![]);
        assert!(ptr::eq(lhs.hardware(), &hw));
        assert!(ptr::eq(rhs.hardware(), &hw));
        assert!(ptr::eq(ret.hardware(), &hw));

        assert_eq!(ret.calculate().unwrap().get_scalar_f32(), Ok(0.5));
    }

    #[test]
    fn test_fill_scalar() {
        let hw = RefCell::new(CpuHardware::new());
        let g = RefCell::new(Graph::new());
        let ret = Node::fill(&hw, &g, make_shape![], 123.);
        assert_eq!(ret.shape(), make_shape![]);
        assert!(ptr::eq(ret.hardware(), &hw));
        assert_eq!(ret.calculate().unwrap().get_scalar_f32(), Ok(123.));
    }

    #[test]
    fn test_fill_0() {
        let hw = RefCell::new(CpuHardware::new());
        let g = RefCell::new(Graph::new());
        let ret = Node::fill(&hw, &g, make_shape![0], 123.);
        assert_eq!(ret.shape(), make_shape![0]);
        assert!(ptr::eq(ret.hardware(), &hw));
        assert_eq!(ret.calculate().unwrap().get_values_f32(), vec![]);
    }

    #[test]
    fn test_fill_n() {
        let hw = RefCell::new(CpuHardware::new());
        let g = RefCell::new(Graph::new());
        let ret = Node::fill(&hw, &g, make_shape![3], 123.);
        assert_eq!(ret.shape(), make_shape![3]);
        assert!(ptr::eq(ret.hardware(), &hw));
        assert_eq!(
            ret.calculate().unwrap().get_values_f32(),
            vec![123., 123., 123.]
        );
    }

    #[test]
    fn test_multiple_computation() {
        let hw = RefCell::new(CpuHardware::new());
        let g = RefCell::new(Graph::new());

        let a = Node::from_scalar(&hw, &g, 1.);
        let b = Node::from_scalar(&hw, &g, 2.);
        let c = Node::from_scalar(&hw, &g, 3.);
        let y = a + -b * c;

        assert_eq!(a.shape(), make_shape![]);
        assert_eq!(b.shape(), make_shape![]);
        assert_eq!(c.shape(), make_shape![]);
        assert_eq!(y.shape(), make_shape![]);
        assert!(ptr::eq(a.hardware(), &hw));
        assert!(ptr::eq(b.hardware(), &hw));
        assert!(ptr::eq(c.hardware(), &hw));
        assert!(ptr::eq(y.hardware(), &hw));

        assert_eq!(y.calculate().unwrap().get_scalar_f32(), Ok(-5.));
    }

    #[test]
    fn test_grad_self() {
        let hw = RefCell::new(CpuHardware::new());
        let g = RefCell::new(Graph::new());

        let x = Node::from_scalar(&hw, &g, 42.);

        let gx = grad(x, &[x]).unwrap();
        assert_eq!(gx.len(), 1);

        assert_eq!(gx[0].shape(), make_shape![]);
        assert!(ptr::eq(gx[0].hardware(), &hw));
        assert_eq!(gx[0].calculate().unwrap().get_scalar_f32(), Ok(1.));
    }

    #[test]
    fn test_grad_unrelated() {
        let hw = RefCell::new(CpuHardware::new());
        let g = RefCell::new(Graph::new());

        let x = Node::from_scalar(&hw, &g, 42.);
        let y = Node::from_scalar(&hw, &g, 42.);

        let gx = grad(y, &[x]).unwrap();
        assert_eq!(gx.len(), 1);

        assert_eq!(gx[0].shape(), make_shape![]);
        assert!(ptr::eq(gx[0].hardware(), &hw));
        assert_eq!(gx[0].calculate().unwrap().get_scalar_f32(), Ok(0.));
    }

    #[test]
    fn test_grad_neg() {
        let hw = RefCell::new(CpuHardware::new());
        let g = RefCell::new(Graph::new());

        let x = Node::from_scalar(&hw, &g, 42.);
        let y = -x;

        let gx = grad(y, &[x]).unwrap();
        assert_eq!(gx.len(), 1);

        assert_eq!(gx[0].shape(), make_shape![]);
        assert!(ptr::eq(gx[0].hardware(), &hw));
        assert_eq!(gx[0].calculate().unwrap().get_scalar_f32(), Ok(-1.));
    }

    #[test]
    fn test_grad_add() {
        let hw = RefCell::new(CpuHardware::new());
        let g = RefCell::new(Graph::new());

        let a = Node::from_scalar(&hw, &g, 123.);
        let b = Node::from_scalar(&hw, &g, 456.);
        let y = a + b;

        let gx = grad(y, &[a, b]).unwrap();
        assert_eq!(gx.len(), 2);

        assert_eq!(gx[0].shape(), make_shape![]);
        assert!(ptr::eq(gx[0].hardware(), &hw));
        assert_eq!(gx[0].calculate().unwrap().get_scalar_f32(), Ok(1.));

        assert_eq!(gx[1].shape(), make_shape![]);
        assert!(ptr::eq(gx[1].hardware(), &hw));
        assert_eq!(gx[1].calculate().unwrap().get_scalar_f32(), Ok(1.));
    }

    #[test]
    fn test_grad_sub() {
        let hw = RefCell::new(CpuHardware::new());
        let g = RefCell::new(Graph::new());

        let a = Node::from_scalar(&hw, &g, 123.);
        let b = Node::from_scalar(&hw, &g, 456.);
        let y = a - b;

        let gx = grad(y, &[a, b]).unwrap();
        assert_eq!(gx.len(), 2);

        assert_eq!(gx[0].shape(), make_shape![]);
        assert!(ptr::eq(gx[0].hardware(), &hw));
        assert_eq!(gx[0].calculate().unwrap().get_scalar_f32(), Ok(1.));

        assert_eq!(gx[1].shape(), make_shape![]);
        assert!(ptr::eq(gx[1].hardware(), &hw));
        assert_eq!(gx[1].calculate().unwrap().get_scalar_f32(), Ok(-1.));
    }

    #[test]
    fn test_grad_mul() {
        let hw = RefCell::new(CpuHardware::new());
        let g = RefCell::new(Graph::new());

        let a = Node::from_scalar(&hw, &g, 123.);
        let b = Node::from_scalar(&hw, &g, 456.);
        let y = a * b;

        let gx = grad(y, &[a, b]).unwrap();
        assert_eq!(gx.len(), 2);

        assert_eq!(gx[0].shape(), make_shape![]);
        assert!(ptr::eq(gx[0].hardware(), &hw));
        assert_eq!(gx[0].calculate().unwrap().get_scalar_f32(), Ok(456.));

        assert_eq!(gx[1].shape(), make_shape![]);
        assert!(ptr::eq(gx[1].hardware(), &hw));
        assert_eq!(gx[1].calculate().unwrap().get_scalar_f32(), Ok(123.));
    }

    #[test]
    fn test_grad_div() {
        let hw = RefCell::new(CpuHardware::new());
        let g = RefCell::new(Graph::new());

        let a = Node::from_scalar(&hw, &g, 3.);
        let b = Node::from_scalar(&hw, &g, 2.);
        let y = a / b;

        let gx = grad(y, &[a, b]).unwrap();
        assert_eq!(gx.len(), 2);

        assert_eq!(gx[0].shape(), make_shape![]);
        assert!(ptr::eq(gx[0].hardware(), &hw));
        // dy/da == 1/b
        assert_eq!(gx[0].calculate().unwrap().get_scalar_f32(), Ok(0.5));

        assert_eq!(gx[1].shape(), make_shape![]);
        assert!(ptr::eq(gx[1].hardware(), &hw));
        // dy/db == -a/b^2
        assert_eq!(gx[1].calculate().unwrap().get_scalar_f32(), Ok(-0.75));
    }
}

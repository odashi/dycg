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
/// * `Ok(Vec<Node>)` - New `Node`s representing the derivative dy/dx. The order of elements
///   corresponds to that of `x`.
/// * `Err(Error)` - Some errors occurred during the process.
pub fn grad<'hw, 'op, 'g>(
    _y: Node<'hw, 'op, 'g>,
    _x: &[Node<'hw, 'op, 'g>],
) -> Result<Vec<Node<'hw, 'op, 'g>>> {
    Err(Error::NotSupported("Not implemented.".to_string()))
}

#[cfg(test)]
mod tests {
    use crate::graph::Graph;
    use crate::hardware::cpu::CpuHardware;
    use crate::make_shape;
    use crate::node::Node;
    use std::cell::RefCell;
    use std::ptr;

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
}

use std::fmt;
use std::rc::Rc;

use scoped_tls_hkt::scoped_thread_local;

use crate::error::Error;
use crate::operation::Operation;
use crate::result::Result;
use crate::value::Value;

scoped_thread_local!(static mut CURRENT_GRAPH: Graph);

/// Computation graph.
pub(crate) struct Graph {
    nodes: Vec<Node>,
}

impl Graph {
    pub(crate) fn new() -> Self {
        Self { nodes: vec![] }
    }

    fn check_address(&self, addr: Address) -> Result<()> {
        if addr.node_id >= self.nodes.len() {
            return Err(Error::InvalidAddress(format!(
                "Address {} does not point to a valid node.",
                addr
            )));
        }
        unsafe {
            if addr.output_id
                >= self
                    .nodes
                    .get_unchecked(addr.node_id)
                    .operation
                    .output_size()
            {
                return Err(Error::InvalidAddress(format!(
                    "Address {} points to a valid node, but does not point to a valid output.",
                    addr
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn add_node(
        &mut self,
        operation: Box<dyn Operation>,
        inputs: Vec<Address>,
    ) -> Result<Vec<Address>> {
        let new_node_id = self.nodes.len();
        let input_size = operation.input_size();
        let output_size = operation.output_size();

        if inputs.len() != input_size {
            return Err(Error::InvalidLength(format!(
                "Operation requires {} inputs, but got {}.",
                operation.input_size(),
                inputs.len()
            )));
        }

        if output_size == 0 {
            return Err(Error::OutOfRange(format!(
                "Operation must return at least 1 output."
            )));
        }

        for &addr in &inputs {
            self.check_address(addr)?;
        }

        self.nodes.push(Node {
            operation,
            inputs,
            outputs: None,
        });

        Ok((0..output_size)
            .map(|output_id| Address {
                node_id: new_node_id,
                output_id,
            })
            .collect())
    }

    unsafe fn is_calculated(&self, node_id: usize) -> bool {
        self.nodes.get_unchecked(node_id).outputs.is_some()
    }

    unsafe fn clone_value_unchecked(&self, addr: Address) -> Rc<Value> {
        self.nodes
            .get_unchecked(addr.node_id)
            .outputs
            .as_ref()
            .unwrap()
            .get_unchecked(addr.output_id)
            .clone()
    }

    pub(crate) fn calculate(&mut self, target: Address) -> Result<Rc<Value>> {
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
        let mut action_stack = vec![(target.node_id, Action::Fetch)];

        while !action_stack.is_empty() {
            let (node_id, action) = action_stack.pop().unwrap();
            match action {
                Action::Fetch => {
                    unsafe {
                        // If the node holds a value already, we don't need to do nothing.
                        if self.is_calculated(node_id) {
                            continue;
                        }

                        // We need to perform the operation.
                        action_stack.push((node_id, Action::Perform));

                        // Before performing the operation, we need all inputs to be calculated.
                        for &input_addr in &self.nodes.get_unchecked(node_id).inputs {
                            action_stack.push((input_addr.node_id, Action::Fetch));
                        }
                    }
                }
                Action::Perform => {
                    unsafe {
                        let node = self.nodes.get_unchecked(node_id);

                        // Collect the input values.
                        // At this point, all the inputs should be calculated already.
                        let inputs = node
                            .inputs
                            .iter()
                            .map(|addr| {
                                self.nodes
                                    .get_unchecked(addr.node_id)
                                    .outputs
                                    .as_ref()
                                    .unwrap()
                                    .get_unchecked(addr.output_id)
                                    .as_ref()
                            })
                            .collect::<Vec<&Value>>();

                        // Perform the operation.
                        self.nodes.get_unchecked_mut(node_id).outputs =
                            Some(node.operation.perform(&inputs).unwrap());
                    }
                }
            }
        }

        Ok(unsafe { self.clone_value_unchecked(target) })
    }
}

/// Node in computation graphs, representing an individual operation.
pub(crate) struct Node {
    operation: Box<dyn Operation>,
    inputs: Vec<Address>,
    outputs: Option<Vec<Rc<Value>>>,
}

/// Address in an computation graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Address {
    node_id: usize,
    output_id: usize,
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}:{}", self.node_id, self.output_id)
    }
}

pub(crate) fn trace(callback: impl FnOnce() -> ()) -> () {
    CURRENT_GRAPH.set(&mut Graph::new(), || {
        callback();
    });
}

pub(crate) fn with_current_graph(callback: impl FnOnce(&mut Graph) -> ()) -> () {
    CURRENT_GRAPH.with(callback);
}

#[cfg(test)]
mod tests {
    use crate::graph::*;
    use crate::make_shape;
    use crate::operation::{Add, Constant};
    use crate::value::{make_scalar, Value};

    #[test]
    fn it_works() {
        trace(|| {
            with_current_graph(|g| {
                assert_eq!(g.nodes.len(), 0);

                let lhs = g
                    .add_node(Box::new(Constant::new(make_scalar(1.))), vec![])
                    .unwrap();
                assert_eq!(g.nodes.len(), 1);
                assert_eq!(
                    lhs,
                    vec![Address {
                        node_id: 0,
                        output_id: 0
                    }]
                );

                let rhs = g
                    .add_node(Box::new(Constant::new(make_scalar(2.))), vec![])
                    .unwrap();
                assert_eq!(g.nodes.len(), 2);
                assert_eq!(
                    rhs,
                    vec![Address {
                        node_id: 1,
                        output_id: 0
                    }]
                );

                let ret = g
                    .add_node(Box::new(Add::new()), vec![lhs[0], rhs[0]])
                    .unwrap();
                assert_eq!(g.nodes.len(), 3);
                assert_eq!(
                    ret,
                    vec![Address {
                        node_id: 2,
                        output_id: 0
                    }]
                );

                let retval = g.calculate(ret[0]).unwrap();
                assert_eq!(*retval.shape(), make_shape![]);
                assert_eq!(retval.to_vec(), vec![3.]);
            });
        });
    }
}

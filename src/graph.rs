use scoped_tls_hkt::scoped_thread_local;

use crate::operation::Operation;
use crate::value::Value;

scoped_thread_local!(static mut CURRENT_GRAPH: Graph);

/// Computation graph.
pub(crate) struct Graph {
    nodes: Vec<Node>,
}

impl Graph {
    fn new() -> Self {
        Self { nodes: vec![] }
    }
}

/// Node in computation graphs, representing an individual operation.
struct Node {
    operation: Box<dyn Operation>,
    inputs: Vec<Address>,
    outputs: Vec<Value>,
}

/// Address in an computation graph.
struct Address {
    node_id: usize,
    output_id: usize,
}

pub fn trace(callback: impl FnOnce() -> ()) -> () {
    CURRENT_GRAPH.set(&mut Graph::new(), || {
        callback();
    });
}

pub(crate) fn with_current_graph(callback: impl FnOnce(&mut Graph) -> ()) -> () {
    CURRENT_GRAPH.with(callback);
}

#[cfg(test)]
mod tests {
    use crate::graph::{trace, with_current_graph};

    #[test]
    fn it_works() {
        trace(|| {
            with_current_graph(|g| {
                assert!((*g).nodes.is_empty());
            });
        });
    }
}

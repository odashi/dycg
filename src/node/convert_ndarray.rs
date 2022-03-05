use crate::node::*;

// Implements separate definitions of TryFrom<Array> for ndarray::ArrayN
// for the same reason with that of Array.

macro_rules! define_try_from_node {
    ( $dest:ty ) => {
        impl<'hw: 'op, 'op: 'g, 'g> TryFrom<Node<'hw, 'op, 'g>> for $dest {
            type Error = Error;
            fn try_from(src: Node<'hw, 'op, 'g>) -> Result<Self> {
                Self::try_from(&src.calculate()?)
            }
        }
    };
}

define_try_from_node!(ndarray::Array0<f32>);
define_try_from_node!(ndarray::Array1<f32>);
define_try_from_node!(ndarray::Array2<f32>);
define_try_from_node!(ndarray::Array3<f32>);
define_try_from_node!(ndarray::Array4<f32>);
define_try_from_node!(ndarray::Array5<f32>);
define_try_from_node!(ndarray::Array6<f32>);

#[cfg(test)]
mod tests;

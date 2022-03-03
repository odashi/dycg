use crate::array::*;

// ndarray has separate definitions of Array0 - Array6 although they are based on some generics.
// We also provide separate definitions of TryFrom<&Array> for each ArrayN to fit the same manner.

macro_rules! define_try_from_array {
    ( $n:expr, $dest:ty ) => {
        impl<'hw> TryFrom<&Array<'hw>> for $dest {
            type Error = Error;
            fn try_from(src: &Array<'hw>) -> Result<Self> {
                let shape = src.shape.as_array::<$n>()?;
                Ok(unsafe { Self::from_shape_vec_unchecked(shape, src.get_values_f32()) })
            }
        }
    };
}

define_try_from_array!(0, ndarray::Array0<f32>);
define_try_from_array!(1, ndarray::Array1<f32>);
define_try_from_array!(2, ndarray::Array2<f32>);
define_try_from_array!(3, ndarray::Array3<f32>);
define_try_from_array!(4, ndarray::Array4<f32>);
define_try_from_array!(5, ndarray::Array5<f32>);
define_try_from_array!(6, ndarray::Array6<f32>);

#[cfg(test)]
mod tests;

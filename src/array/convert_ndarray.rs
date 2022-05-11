use crate::array::*;

// ndarray has separate definitions of Array0 - Array6 although they are based on some generics.
// We also provide separate definitions of TryFrom<&Array> and IntoArray for each ArrayN to fit the same manner.

macro_rules! define_into_array {
    ( $src_ty:ty ) => {
        impl IntoArray for $src_ty {
            fn into_array(self, hardware: &RefCell<dyn Hardware>) -> Array {
                Array::constant_f32(
                    hardware,
                    Shape::from_slice(self.shape()),
                    // `Array` supports only data with the row-major order.
                    self.as_standard_layout().as_slice().unwrap(),
                )
                .unwrap()
            }
        }
    };
}

define_into_array!(&ndarray::Array0<f32>);
define_into_array!(&ndarray::Array1<f32>);
define_into_array!(&ndarray::Array2<f32>);
define_into_array!(&ndarray::Array3<f32>);
define_into_array!(&ndarray::Array4<f32>);
define_into_array!(&ndarray::Array5<f32>);
define_into_array!(&ndarray::Array6<f32>);

macro_rules! define_into_array {
    ( $src_ty:ty ) => {
        impl IntoArray for $src_ty {
            fn into_array(self, hardware: &RefCell<dyn Hardware>) -> Array {
                Array::constant_f32(
                    hardware,
                    Shape::from_slice(self.shape()),
                    // `Array` supports only data with the row-major order.
                    self.as_standard_layout().as_slice().unwrap(),
                )
                .unwrap()
            }
        }
    };
}

define_into_array!(&ndarray::Array0<f32>);
define_into_array!(&ndarray::Array1<f32>);
define_into_array!(&ndarray::Array2<f32>);
define_into_array!(&ndarray::Array3<f32>);
define_into_array!(&ndarray::Array4<f32>);
define_into_array!(&ndarray::Array5<f32>);
define_into_array!(&ndarray::Array6<f32>);

macro_rules! define_try_from_array {
    ( $as_array_fn:ident, $dest_ty:ty ) => {
        impl<'hw> TryFrom<&Array<'hw>> for $dest_ty {
            type Error = Error;
            fn try_from(src: &Array<'hw>) -> Result<Self> {
                let shape = src.shape.$as_array_fn()?;
                Ok(unsafe { Self::from_shape_vec_unchecked(shape, src.get_values_f32()) })
            }
        }
    };
}

define_try_from_array!(as_array0, ndarray::Array0<f32>);
define_try_from_array!(as_array1, ndarray::Array1<f32>);
define_try_from_array!(as_array2, ndarray::Array2<f32>);
define_try_from_array!(as_array3, ndarray::Array3<f32>);
define_try_from_array!(as_array4, ndarray::Array4<f32>);
define_try_from_array!(as_array5, ndarray::Array5<f32>);
define_try_from_array!(as_array6, ndarray::Array6<f32>);

#[cfg(test)]
mod tests;

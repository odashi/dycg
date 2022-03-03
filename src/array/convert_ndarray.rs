use crate::array::*;

impl<'hw> TryFrom<&Array<'hw>> for ndarray::Array0<f32> {
    type Error = Error;
    fn try_from(src: &Array<'hw>) -> Result<Self> {
        src.shape().check_is_scalar()?;
        Ok(unsafe { Self::from_shape_vec_unchecked((), src.get_values_f32()) })
    }
}

impl<'hw> TryFrom<&Array<'hw>> for ndarray::Array1<f32> {
    type Error = Error;
    fn try_from(src: &Array<'hw>) -> Result<Self> {
        if let Ok(shape) = src.shape.as_array::<1>() {
            Ok(unsafe { Self::from_shape_vec_unchecked(shape, src.get_values_f32()) })
        } else {
            Err(Error::InvalidShape(format!(
                "Attempted to generate ndarray::Array1 from {}-dimensional Array.",
                src.shape().num_dimensions()
            )))
        }
    }
}

#[cfg(test)]
mod tests;

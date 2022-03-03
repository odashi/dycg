use crate::array::*;

impl<'hw> TryFrom<&Array<'hw>> for ndarray::Array0<f32> {
    type Error = Error;
    fn try_from(src: &Array<'hw>) -> Result<Self> {
        src.shape().check_is_scalar()?;
        Ok(unsafe { Self::from_shape_vec_unchecked((), src.get_values_f32()) })
    }
}

#[cfg(test)]
mod tests;

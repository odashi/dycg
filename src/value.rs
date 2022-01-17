use crate::error::Error;
use crate::make_shape;
use crate::result::Result;
use crate::shape::Shape;

/// Actual value in the computation graph.
#[derive(Debug)]
pub(crate) struct Value {
    shape: Shape,
    values: Vec<f32>,
}

impl Value {
    pub(crate) fn new(shape: Shape, values: Vec<f32>) -> Result<Self> {
        if values.len() != shape.num_values() {
            return Err(Error::InvalidLength(format!(
                "shape required {} values, but got {} values.",
                shape.num_values(),
                values.len(),
            )));
        }
        Ok(Self { shape, values })
    }

    pub(crate) fn shape(&self) -> &Shape {
        &self.shape
    }

    pub(crate) fn to_vec(&self) -> Vec<f32> {
        self.values.clone()
    }
}

pub(crate) fn make_scalar(value: f32) -> Value {
    Value::new(make_shape![], vec![value]).unwrap()
}

#[cfg(test)]
mod tests {
    use crate::make_shape;
    use crate::value::make_scalar;

    #[test]
    fn test_scalar() {
        let value = make_scalar(123.);
        assert_eq!(*value.shape(), make_shape![]);
        assert_eq!(*value.to_vec(), vec![123.]);
    }
}

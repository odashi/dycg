use crate::array::Array;
use crate::error::Error;
use crate::hardware::Hardware;
use crate::operator::Operator;
use crate::result::Result;
use crate::shape::Shape;
use std::cell::RefCell;

/// Placeholder of `Array`s.
/// Unlike `Option`, the object always holds its `Shape` and `Hardware` informatin.
pub(crate) enum ArrayPlaceholder<'hw> {
    /// `Array` is not assigned, while its `Shape` is known.
    Unassigned(Shape, &'hw RefCell<dyn Hardware>),

    /// `Array` is assigned.
    Assigned(Array<'hw>),
}

impl<'hw> ArrayPlaceholder<'hw> {
    /// Obtains the `Shape` of this placeholder.
    ///
    /// # Returns
    ///
    /// A reference to the inner `Shape` object.
    pub(crate) fn shape(&self) -> &Shape {
        match self {
            Self::Unassigned(ref shape, _) => shape,
            Self::Assigned(ref array) => array.shape(),
        }
    }

    /// Obtains the `Hardware` of this placeholder.
    ///
    /// # Returns
    ///
    /// A reference to the `Hardware` object.
    pub(crate) fn hardware(&self) -> &'hw RefCell<dyn Hardware> {
        match self {
            Self::Unassigned(_, hardware) => hardware,
            Self::Assigned(ref array) => array.hardware(),
        }
    }

    /// Obtains the `Array` if the placeholder holds it.
    ///
    /// # Returns
    ///
    /// * `Some(&Array)` - A reference to the inner `Array` object.
    /// * `None` - The placeholder does not hold the `Array` object.
    pub(crate) fn array(&self) -> Option<&Array<'hw>> {
        match self {
            Self::Unassigned(_, _) => None,
            Self::Assigned(ref array) => Some(array),
        }
    }
}

/// Individual step in computation graphs.
/// Step owns an Operator which consumes several values produced by preceding steps.
pub(crate) struct Step<'hw: 'op, 'op> {
    /// Operator owned by this step.
    pub(crate) operator: Box<dyn Operator<'hw> + 'op>,

    /// Input step IDs.
    pub(crate) inputs: Vec<usize>,

    /// Output values.
    pub(crate) output: ArrayPlaceholder<'hw>,
}

impl<'hw: 'op, 'op> Step<'hw, 'op> {
    /// Creates a new `Step` object.
    fn new(
        operator: Box<dyn Operator<'hw> + 'op>,
        inputs: Vec<usize>,
        output: ArrayPlaceholder<'hw>,
    ) -> Self {
        Self {
            operator,
            inputs,
            output,
        }
    }
}

// Computation graph.
pub struct Graph<'hw: 'op, 'op> {
    /// All steps registered to this graph.
    steps: Vec<Step<'hw, 'op>>,
}

impl<'hw: 'op, 'op> Graph<'hw, 'op> {
    /// Creates a new empty `Graph` object.
    ///
    /// # Returns
    ///
    /// A new `Graph` object containing zero steps.
    pub fn new() -> Self {
        Self { steps: vec![] }
    }

    /// Returns the number of registered steps.
    ///
    /// # Returns
    ///
    /// The number of registered operations.
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }

    /// Returns a reference to the specified `Step`.
    ///
    /// # Arguments
    ///
    /// * `step_id` - Step ID to be checked.
    ///
    /// # Returns
    ///
    /// * `Some(&Step)` - A reference to the specified `Step` in this graph.
    /// * `None` - `step_id` is invalid.
    pub(crate) fn get_step(&self, step_id: usize) -> Result<&Step<'hw, 'op>> {
        self.steps
            .get(step_id)
            .ok_or_else(|| Error::InvalidNode(format!("Invalid step ID: {}", step_id)))
    }

    /// Inserts a new step into this graph.
    ///
    /// # Arguments
    ///
    /// * `operator` - `Operator` for the new step.
    /// * `inputs` - Input steps.
    ///
    /// # Returns
    ///
    /// * `Ok(usize)` - The new step ID inserted by this function.
    /// * `Err(Error)` - Some error occurred during the process.
    pub(crate) fn add_step<'g>(
        &'g mut self,
        operator: Box<dyn Operator<'hw> + 'op>,
        inputs: Vec<usize>,
    ) -> Result<usize> {
        let new_step_id = self.steps.len();
        let input_size = operator.input_size();

        if inputs.len() != input_size {
            return Err(Error::InvalidLength(format!(
                "Operator requires {} inputs, but got {}.",
                operator.input_size(),
                inputs.len()
            )));
        }

        let input_steps = inputs
            .iter()
            .map(|&step_id| self.get_step(step_id))
            .collect::<Result<Vec<_>>>()?;
        let (input_shapes, input_hardwares): (Vec<_>, Vec<_>) = input_steps
            .iter()
            .map(|&step| (step.output.shape(), step.output.hardware()))
            .unzip();

        let output_shape = operator.perform_shape(&input_shapes)?;
        let output_hardware = operator.perform_hardware(&input_hardwares)?;

        self.steps.push(Step::new(
            operator,
            inputs,
            ArrayPlaceholder::Unassigned(output_shape, output_hardware),
        ));

        Ok(new_step_id)
    }

    /// Performs calculation to obtain the value of specified node.
    ///
    /// This function internally performs a push-down automaton to recursively obtain the values
    /// necessary to calculate the target node.
    /// If nodes (both target/parents) are already calculated, calculation is skipped and the cached
    /// values are used instead.
    ///
    /// # Arguments
    ///
    /// * `target` - Target step to obtain the value.
    ///
    /// # Returns
    ///
    /// * Calculated/cached value associated to `target`.
    pub(crate) fn calculate(&mut self, target: usize) -> Result<Array<'hw>> {
        // Avoiding an edge case: inner step_ids should be correct, but `target` is not constrained.
        if target >= self.steps.len() {
            return Err(Error::InvalidNode(format!("Invalid step ID: {}", target)));
        }

        // Actions for the push-down automaton representing the following procedure:
        /*
            fn calculate(step) {
                if assigned(step) { return; }
                for input in step.inputs { calculate(input); }
                perform(step);
            }
        */
        enum Action {
            Fetch,
            Perform,
        }

        // action_stack represents the state of the push-down automaton.
        let mut action_stack = vec![(target, Action::Fetch)];

        while !action_stack.is_empty() {
            let (step_id, action) = action_stack.pop().unwrap();
            match action {
                Action::Fetch => {
                    let step = unsafe { self.steps.get_unchecked(step_id) };

                    // If the node holds a value already, we need to do nothing.
                    if let ArrayPlaceholder::Assigned(_) = step.output {
                        continue;
                    }

                    // We need to perform the operator.
                    action_stack.push((step_id, Action::Perform));

                    // Before performing the operator, we need all inputs to be calculated.
                    for &input in &step.inputs {
                        action_stack.push((input, Action::Fetch));
                    }
                }
                Action::Perform => {
                    let step = unsafe { self.steps.get_unchecked(step_id) };

                    // Collect the input values.
                    // At this point, all the inputs should be calculated already.
                    let inputs = step
                        .inputs
                        .iter()
                        .map(|&input_id| {
                            unsafe { self.steps.get_unchecked(input_id) }
                                .output
                                .array()
                                .unwrap()
                        })
                        .collect::<Vec<_>>();

                    // Perform the operator.
                    unsafe { self.steps.get_unchecked_mut(step_id) }.output =
                        ArrayPlaceholder::Assigned(step.operator.perform(&inputs).unwrap());
                }
            }
        }

        if let ArrayPlaceholder::Assigned(ref array) =
            unsafe { self.steps.get_unchecked(target) }.output
        {
            return Ok(array.clone());
        }

        panic!("Target step could not be calculated for some reason.");
    }
}

impl<'hw: 'op, 'op> Default for Graph<'hw, 'op> {
    fn default() -> Self {
        Self::new()
    }
}

// TODO(odashi): add tests.

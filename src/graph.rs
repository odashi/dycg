use crate::array::Array;
use crate::error::Error;
use crate::operator::Operator;
use crate::result::Result;

//// Individual step in computation graphs.
/// Step owns an Operator which consumes several values produced by preceding steps.
pub(crate) struct Step<'hw: 'op, 'op> {
    /// Operator owned by this step.
    pub(crate) operator: Box<dyn Operator<'hw> + 'op>,

    /// Input step IDs.
    pub(crate) inputs: Vec<usize>,

    /// Output values.
    pub(crate) output: Option<Array<'hw>>,
}

impl<'hw: 'op, 'op> Step<'hw, 'op> {
    /// Creates a new `Step` object.
    fn new(
        operator: Box<dyn Operator<'hw> + 'op>,
        inputs: Vec<usize>,
        output: Option<Array<'hw>>,
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
            .ok_or(Error::InvalidNode(format!("Invalid step ID: {}", step_id)))
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

        for &input_id in &inputs {
            self.get_step(input_id)?;
        }

        self.steps.push(Step::new(operator, inputs, None));

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
        let mut action_stack = vec![(target, Action::Fetch)];

        while !action_stack.is_empty() {
            let (step_id, action) = action_stack.pop().unwrap();
            match action {
                Action::Fetch => {
                    let step = self.get_step(step_id)?;

                    // If the node holds a value already, we need to do nothing.
                    if step.output.is_some() {
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
                    let step = self.get_step(step_id)?;

                    // Collect the input values.
                    // At this point, all the inputs should be calculated already.
                    let inputs = step
                        .inputs
                        .iter()
                        .map(|&input_id| self.get_step(input_id).unwrap().output.as_ref().unwrap())
                        .collect::<Vec<_>>();

                    // Perform the operator.
                    self.steps[step_id].output = Some(step.operator.perform(&inputs).unwrap());
                }
            }
        }

        Ok(self
            .get_step(target)
            .unwrap()
            .output
            .as_ref()
            .unwrap()
            .clone())
    }
}

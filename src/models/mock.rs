use crate::model::{Model, Output, ModelPtr};
use crate::batch::Batch;
use std::sync::Arc;

/// Mock model for testing, produces dummy outputs.
pub struct MockModel {}

impl Model for MockModel {
    fn new(p: Option<&std::path::Path>) -> ModelPtr {
        Arc::new(MockModel {})
    }

    fn execute(&self, b: &Batch) -> Output {
        let mut policy = Vec::new();
        policy.resize(b.get_size() * 4096, 1.0 / 4096.0);

        let mut value = Vec::new();
        value.resize(b.get_size(), 0.0);

        Output::new(policy, value)
    }

    fn train(&mut self, b: &crate::train::TrainBatch) {}
    fn write(&self, p: &std::path::Path) {}
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::position::Position;

    /// Tests the mock model can be initialized.
    #[test]
    fn mock_model_can_init() {
        MockModel::new(None);
    }

    /// Tests the mock model produces mock outputs.
    #[test]
    fn mock_model_can_execute() {
        let m = MockModel::new(None);
        let mut b = Batch::new(4);

        b.add(&Position::new(), 0);

        let output = m.execute(&b);

        assert_eq!(output.get_policy(0), &[1.0 / 4096.0; 4096]);
        assert_eq!(output.get_value(0), 0.0);
    }
}
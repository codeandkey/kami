use crate::input::{batch::Batch, trainbatch::TrainBatch};
use crate::model::{Model, Output};
use rand::prelude::*;
use std::error::Error;
use std::path::Path;

/// Mock model for testing, produces dummy outputs.
pub struct MockModel {}

impl Model for MockModel {
    fn read(_: &Path) -> Result<Self, Box<dyn Error>> {
        MockModel::generate()
    }

    fn generate() -> Result<Self, Box<dyn Error>> {
        Ok(MockModel {})
    }

    fn execute(&self, b: &Batch) -> Output {
        let mut policy = Vec::new();
        let mut rng = rand::thread_rng();
        policy.resize_with(b.get_size() * 4096, || {
            rng.next_u32() as f32 / u32::MAX as f32
        });

        let mut value = Vec::new();
        value.resize_with(b.get_size(), || {
            (rng.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
        });

        Output::new(policy, value)
    }

    fn train(&mut self, _: Vec<TrainBatch>) {}
    fn write(&self, _: &Path) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
    fn get_type(&self) -> &'static str {
        "mock"
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::position::Position;

    /// Tests the mock model can be generated.
    #[test]
    fn mock_model_can_generate() {
        MockModel::generate().expect("gen failed");
    }

    /// Tests the mock model can be loaded.
    #[test]
    fn mock_model_can_read() {
        let path = tempfile::tempdir()
            .expect("failed to gen tempdir")
            .into_path();

        MockModel::generate()
            .expect("gen failed")
            .write(&path)
            .expect("write failed");
        MockModel::read(&path).expect("load failed");
    }

    /// Tests the mock model produces mock outputs.
    #[test]
    fn mock_model_can_execute() {
        let m = MockModel::generate().expect("model gen failed");
        let mut b = Batch::new(4);

        b.add(&Position::new());

        let output = m.execute(&b);

        assert_eq!(output.get_policy(0).len(), 4096);
    }

    /// Tests the mock model can write. (to nothing)
    #[test]
    fn mock_model_can_write() {
        let path = tempfile::tempdir()
            .expect("failed to gen tempdir")
            .into_path();

        MockModel::generate()
            .expect("model gen failed")
            .write(&path.join("mock_model_test"))
            .expect("write failed");
    }

    /// Tests the mock model can write.
    #[test]
    fn mock_model_can_train() {
        MockModel::generate()
            .expect("model gen failed")
            .train(vec![TrainBatch::new(1)]);
    }

    /// Tests the mock model can return its type.
    #[test]
    fn mock_model_can_get_type() {
        assert_eq!(
            MockModel::generate().expect("model gen failed").get_type(),
            "mock".to_string()
        );
    }
}

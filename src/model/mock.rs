use crate::model::{Model, ModelPtr, Output};
use crate::input::{
    trainbatch::TrainBatch,
    batch::Batch,
};
use std::io::{self, Write};
use std::fs::File;
use std::path::Path;
use std::sync::{Arc, RwLock};
use rand::prelude::*;

/// Mock model for testing, produces dummy outputs.
pub struct MockModel {}

impl Model for MockModel {
    fn new(_: &Path) -> ModelPtr {
        Arc::new(RwLock::new(MockModel {}))
    }

    fn execute(&self, b: &Batch) -> Output {
        let mut policy = Vec::new();
        let mut rng = rand::thread_rng();
        policy.resize_with(b.get_size() * 4096, || rng.next_u32() as f32 / u32::MAX as f32);

        let mut value = Vec::new();
        value.resize_with(b.get_size(), || (rng.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0);

        Output::new(policy, value)
    }

    fn train(&mut self, _: Vec<TrainBatch>) {}

    fn write(&self, p: &Path) -> Result<(), io::Error> {
        File::open(p)?.write(b"mock")?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::position::Position;
    use std::path::PathBuf;

    /// Tests the mock model can be initialized.
    #[test]
    fn mock_model_can_init() {
        MockModel::new(&PathBuf::from("."));
    }

    /// Tests the mock model produces mock outputs.
    #[test]
    fn mock_model_can_execute() {
        let m = MockModel::new(&PathBuf::from("."));
        let mut b = Batch::new(4);

        b.add(&Position::new());

        let output = m.read().unwrap().execute(&b);

        assert_eq!(output.get_policy(0).len(), 4096);
    }
}

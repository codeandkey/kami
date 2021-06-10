/// Represents a batch for expanding nodes on the tree.
/// This batch tracks multiple inputs like a normal batch but also includes node IDs to expand at.
use crate::input::batch::Batch;
use crate::position::Position;

pub struct TreeBatch {
    selected: Vec<usize>,
    inner: Batch,
}

impl TreeBatch {
    /// Returns a new batch instance with <reserve_size> preallocated space.
    pub fn new(reserve_size: usize) -> Self {
        let mut selected = Vec::new();
        selected.reserve(reserve_size);

        TreeBatch {
            inner: Batch::new(reserve_size),
            selected: selected,
        }
    }

    /// Adds a position snapshot to the batch.
    pub fn add(&mut self, p: &Position, idx: usize) {
        // Store node identifier
        self.selected.push(idx);

        // Add input to inner batch
        self.inner.add(p);
    }

    /// Gets the node index for the <idx>-th position in this batch.
    pub fn get_selected(&self, idx: usize) -> usize {
        self.selected[idx]
    }

    /// Gets a reference to the inner batch.
    pub fn get_inner(&self) -> &Batch {
        &self.inner
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// Tests the treebatch can be initialized.
    #[test]
    fn treebatch_can_initialize() {
        TreeBatch::new(16);
    }

    /// Tests the treebatch can be added to.
    #[test]
    fn treebatch_can_add() {
        let mut b = TreeBatch::new(16);
        b.add(&Position::new(), 0);
    }

    /// Tests the selected nodes can be returned.
    #[test]
    fn treebatch_can_get_selected() {
        let mut b = TreeBatch::new(16);

        b.add(&Position::new(), 0);
        b.add(&Position::new(), 1);
        b.add(&Position::new(), 2);

        assert_eq!(b.get_selected(0), 0);
        assert_eq!(b.get_selected(1), 1);
        assert_eq!(b.get_selected(2), 2);
    }

    /// Tests the inner batch can be returned.
    #[test]
    fn treebatch_can_get_inner() {
        let mut b = TreeBatch::new(16);

        b.add(&Position::new(), 0);
        b.add(&Position::new(), 1);
        b.add(&Position::new(), 2);

        assert_eq!(b.get_inner().get_size(), 3);
    }
}

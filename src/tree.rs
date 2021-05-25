/**
 * Threadsafe tree container. If cloned, still points to the same tree structure.
 */

use crate::node::Node;
use serde::ser::{Serialize, Serializer, SerializeSeq};
use std::sync::{Arc, RwLock, RwLockReadGuard};

#[derive(Clone)]
pub struct Tree {
    nodes: Arc<RwLock<Vec<Node>>>,
}

impl Tree {
    pub fn new() -> Self {
        Tree {
            nodes: Arc::new(RwLock::new(vec![Node::root()])),
        }
    }

    pub fn add(&mut self, n: Node) -> usize {
        let mut wlock = self.nodes.write().unwrap();
        
        wlock.push(n);
        wlock.len()
    }

    pub fn nodes(&self) -> RwLockReadGuard<Vec<Node>> {
        self.nodes.read().unwrap()
    }

    pub fn uct_pairs(&mut self, idx: usize) -> Vec<(usize, f32)> {
        const EXPLORATION: f32 = 1.414;

        let parent_n: u32;
        let parent_children: Vec<usize>;

        {
            let parent = self.nodes()[idx];
            parent_n = parent.get_value().n;
            parent_children = parent.get_children().to_vec();
        }

        let mut pairs = parent_children
            .iter()
            .map(|&cidx| {
                let (value_n, value_w) = self.nodes()[cidx].get_value().both();
                let uct = (value_w / (value_n as f32)) + EXPLORATION * ((parent_n as f32).ln() / (value_n as f32 + 1.0)).sqrt();

                (cidx, uct)
            })
            .collect::<Vec<(usize, f32)>>();

        pairs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        pairs
    }

    pub fn backprop(&mut self, idx: usize, value: f32) {
        self.nodes()[idx].add(value);

        if let Some(pidx) = self.nodes()[idx].get_parent() {
            self.backprop(pidx, -value);
        }
    }
}

impl Serialize for Tree {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where 
        S: Serializer
    {
        let mut state = serializer.serialize_seq(None)?;

        let results: Vec<Result<(), S::Error>> = self.nodes()[0]
            .get_children()
            .iter()
            .map(|x| state.serialize_element(x))
            .collect();

        for res in results {
            res?;
        }

        state.end()
    }
}
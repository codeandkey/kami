/**
 * Search worker.
 */

use crate::batch::Batch;
use crate::net::Model;
use crate::tree::Tree;
use crate::position::Position;

use std::sync::{
    Arc, RwLock,
    atomic::{
        AtomicBool,
        Ordering
    }
};

use serde::Serialize;
use std::thread::{spawn, JoinHandle};

#[derive(Serialize, Clone)]
pub struct Status {
    total_nodes: usize,
    batch_sizes: Vec<usize>,
    state: String,
}

impl Status {
    pub fn set_state(&mut self, state: String) {
        self.state = state;
    }
}

impl Status {
    pub fn new() -> Self {
        Status {
            total_nodes: 0,
            batch_sizes: Vec::new(),
            state: "building".to_string(),
        }
    }
}

pub struct Worker {
    thr: Option<JoinHandle<()>>,
    stopflag: Arc<AtomicBool>,
    tree: Tree,
    status: Arc<RwLock<Status>>,
    rootpos: Position,
    batch_size: usize,
    network: Arc<Model>,
}

impl Worker {
    pub fn new(dst_tree: Tree, stopflag: Arc<AtomicBool>, bsize: usize, rootpos: Position, network: Arc<Model>) -> Worker {
        Worker {
            thr: None,
            tree: dst_tree,
            status: Arc::new(RwLock::new(Status::new())),
            stopflag: stopflag,
            batch_size: bsize,
            rootpos: rootpos,
            network: network,
        }
    }

    pub fn start(&mut self) {
        let mut thr_tree = self.tree.clone();
        let thr_status = self.status.clone();
        let thr_stopflag = self.stopflag.clone();
        let thr_batchsize = self.batch_size;
        let thr_network = self.network.clone();
        let mut thr_position = self.rootpos.clone();

        self.thr = Some(spawn(move || {
            while !thr_stopflag.load(Ordering::Relaxed) {
                // Continue working.

                // (1) Multi-selection - identification and claiming of target nodes
                thr_status.write().unwrap().set_state("building".to_string());

                let next_batch = Batch::build(thr_batchsize, &mut thr_position, &mut thr_tree);

                // (2) Execution - Inputs fed to model and processed

                thr_status.write().unwrap().set_state("execute ".to_string());
                let results = thr_network.execute(&next_batch).expect("Network execute failed!");

                // (3) Backpropagation - Tree updated with results from network, all claimed nodes are expanded

                thr_status.write().unwrap().set_state("backprop".to_string());
                next_batch.apply_results(results, &mut thr_tree);
            }
        }))
    }

    pub fn get_status(&self) -> Status {
        self.status.read().unwrap().clone()
    }

    pub fn join(self) {
        self.thr.unwrap().join().expect("worker join failed");
    }
}
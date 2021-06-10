/**
 * Search worker.
 */
use crate::model::ModelPtr;
use crate::tree::{BatchResponse, TreeReq};

use serde::{Deserialize, Serialize};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, RwLock};
use std::thread::{spawn, JoinHandle};

#[derive(Deserialize, Serialize, Clone)]
pub struct Status {
    pub total_nodes: usize,
    pub batch_sizes: Vec<usize>,
    pub state: String,
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
            state: "idle".to_string(),
        }
    }
}

pub struct Worker {
    thr: Option<JoinHandle<()>>,
    status: Arc<RwLock<Status>>,
    tree_tx: Sender<TreeReq>,
    network: ModelPtr,
}

impl Worker {
    pub fn new(tree_tx: Sender<TreeReq>, network: ModelPtr) -> Worker {
        Worker {
            thr: None,
            status: Arc::new(RwLock::new(Status::new())),
            tree_tx: tree_tx,
            network: network,
        }
    }

    pub fn start(&mut self) {
        let thr_tree_tx = self.tree_tx.clone();
        let thr_status = self.status.clone();
        let thr_network = self.network.clone();

        self.thr = Some(spawn(move || {
            let (tmp_tree_tx, tmp_tree_rx) = channel();

            loop {
                // Continue working.

                // (1) Request batch from tree thread
                thr_status
                    .write()
                    .unwrap()
                    .set_state("requesting batch".to_string());

                if thr_tree_tx
                    .send(TreeReq::BuildBatch(tmp_tree_tx.clone()))
                    .is_err()
                {
                    break;
                }

                // (2) Wait for batch response
                let next_batch = match tmp_tree_rx.recv() {
                    Ok(resp) => match resp {
                        BatchResponse::NextBatch(b) => b,
                        BatchResponse::Stop => break,
                    },
                    Err(_) => break,
                };

                // (3) Execution - Inputs fed to model and processed
                thr_status
                    .write()
                    .unwrap()
                    .set_state("executing".to_string());
                let results = thr_network.read().unwrap().execute(next_batch.get_inner());

                // (4) Backpropagation - send results back to tree
                thr_status
                    .write()
                    .unwrap()
                    .set_state("backprop".to_string());

                // (5) Update status fields
                thr_status.write().unwrap().total_nodes += next_batch.get_inner().get_size();
                thr_status
                    .write()
                    .unwrap()
                    .batch_sizes
                    .push(next_batch.get_inner().get_size());

                if thr_tree_tx
                    .send(TreeReq::Expand(Box::new(results), next_batch))
                    .is_err()
                {
                    break;
                }
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

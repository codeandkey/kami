/// Structures for search worker threads.
use crate::model::{self, Model};
use crate::tree::{BatchResponse, TreeReq};

use serde::{Deserialize, Serialize};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, RwLock};
use std::thread::{spawn, JoinHandle};

/// Describes the status of a worker thread.
#[derive(Deserialize, Serialize, Clone)]
pub struct Status {
    pub total_nodes: usize,
    pub batch_sizes: Vec<usize>,
    pub state: String,
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

/// Manages a search thread.
pub struct Worker {
    thr: Option<JoinHandle<()>>,
    status: Arc<RwLock<Status>>,
    tree_tx: Sender<TreeReq>,
    network: Arc<Model>,
}

impl Worker {
    /// Returns a new worker instance. Does not start the thread immediately.
    pub fn new(tree_tx: Sender<TreeReq>, network: Arc<Model>) -> Worker {
        Worker {
            thr: None,
            status: Arc::new(RwLock::new(Status::new())),
            tree_tx: tree_tx,
            network: network,
        }
    }

    /// Starts the worker thread.
    pub fn start(&mut self) {
        let thr_tree_tx = self.tree_tx.clone();
        let thr_status = self.status.clone();
        let thr_network = self.network.clone();

        self.thr = Some(spawn(move || {
            let (tmp_tree_tx, tmp_tree_rx) = channel();

            loop {
                // Continue working.

                // (1) Request batch from tree thread
                thr_status.write().unwrap().state = "requesting batch".to_string();

                if thr_tree_tx
                    .send(TreeReq::BuildBatch(tmp_tree_tx.clone()))
                    .is_err()
                {
                    break;
                }

                // (2) Wait for batch response
                let (next_batch, nodes) = match tmp_tree_rx.recv() {
                    Ok(resp) => match resp {
                        BatchResponse::NextBatch(b, terminals) => (b, terminals),
                        BatchResponse::Stop => break,
                    },
                    Err(_) => break,
                };

                // Add terminal evaluations to node count
                thr_status.write().unwrap().total_nodes += nodes;

                // (3) Execution - Inputs fed to model and processed
                thr_status.write().unwrap().state = "executing".to_string();

                if next_batch.get_inner().get_size() > 0 {
                    let results = model::execute(&thr_network, next_batch.get_inner());

                    // Update status fields
                    thr_status
                        .write()
                        .unwrap()
                        .batch_sizes
                        .push(next_batch.get_inner().get_size());

                    // (4) Backpropagation - send results back to tree
                    thr_status.write().unwrap().state = "backprop".to_string();

                    if thr_tree_tx
                        .send(TreeReq::Expand(Box::new(results), next_batch))
                        .is_err()
                    {
                        break;
                    }
                }
            }
        }))
    }

    /// Returns the status of this worker.
    pub fn get_status(&self) -> Status {
        self.status.read().unwrap().clone()
    }

    /// Waits for this worker to join.
    /// The worker must be terminated by sending a RequestStop to the associated tree.
    pub fn join(self) {
        self.thr.unwrap().join().expect("worker join failed");
    }
}

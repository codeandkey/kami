// Worker structure.

use crate::batch::{Batch, BatchResult};
use crate::model::Model;

use std::thread::{spawn, JoinHandle};
use std::sync::{
    Arc,
    mpsc::{channel, Sender}
};

/// Outgoing messages from workers.
pub enum WorkerMsg {
    Ready(Sender<Option<Box<Batch>>>),
    Expand(Box<BatchResult>),
}

/// Manages a single worker thread.
pub struct Worker {
    handle: JoinHandle<()>,
    tx: Sender<Option<Box<Batch>>>,
}

impl Worker {
    /// Starts a new worker thread.
    pub fn new(outgoing: Sender<WorkerMsg>, model: Arc<Model>) -> Self {
        let (tx, rx) = channel();
        let thr_tx = tx.clone();

        let handle = spawn(move || {
            loop {
                outgoing.send(WorkerMsg::Ready(thr_tx.clone())).unwrap();

                match rx.recv().unwrap() {
                    Some(nb) => {
                        let res = model.execute(*nb);
                        outgoing.send(WorkerMsg::Expand(Box::new(res))).unwrap();
                    },
                    None => break
                }
            }
            
        });

        Worker {
            handle: handle,
            tx: tx,
        }
    }

    /// Stops and joins this worker.
    pub fn join(self) {
        self.tx.send(None).unwrap();
        self.handle.join().expect("worker join failed");
    }
}
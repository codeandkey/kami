use crate::net::Model;
use crate::position::Position;
use crate::tree::{Tree, TreeReq};
use crate::worker::{self, Worker};

use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{spawn, JoinHandle};
use std::time::Duration;

#[derive(Clone, Serialize, Deserialize)]
pub struct Status {
    status: String,
    workers: Vec<worker::Status>,
}

pub struct Searcher {
    workers: Arc<Mutex<Vec<Worker>>>,
    stopflag: Arc<RwLock<bool>>,
    status_string: Arc<RwLock<String>>,
    handle: Option<JoinHandle<()>>,
}

impl Searcher {
    pub fn new() -> Self {
        Searcher {
            workers: Arc::new(Mutex::new(Vec::new())),
            stopflag: Arc::new(RwLock::new(false)),
            handle: None,
            status_string: Arc::new(RwLock::new("idle".to_string())),
        }
    }

    pub fn start(
        &mut self,
        thr_stime: Option<usize>,
        model: Arc<Model>,
        thr_pos: Position,
    ) -> bool {
        if self.handle.is_some() {
            println!("Search is already running!");
            return false;
        }

        *self.stopflag.write().unwrap() = false;

        let thr_stopflag = self.stopflag.clone();
        let thr_model = model.clone();
        let thr_status_string = self.status_string.clone();
        let thr_workers = self.workers.clone();

        println!("Starting search on {}", thr_pos.get_fen());

        self.handle = Some(spawn(move || {
            let (thr_tree_tx, thr_tree_handle) = Tree::new(thr_pos).run();

            for _ in 0..num_cpus::get() {
                let mut new_worker =
                    Worker::new(thr_stopflag.clone(), thr_tree_tx.clone(), thr_model.clone());
                new_worker.start();
                thr_workers.lock().unwrap().push(new_worker);
            }

            *thr_status_string.write().unwrap() = "searching".to_string();

            // Sleep if on timer, otherwise wait for manual stop
            match thr_stime {
                Some(t) => {
                    std::thread::sleep(Duration::from_millis(t as u64));
                }
                None => {
                    while !*thr_stopflag.read().unwrap() {
                        std::thread::sleep(Duration::from_millis(25));
                    }
                }
            }

            const TIMESTEP: u64 = 25;
            let mut elapsed: u64 = 0;

            while !*thr_stopflag.read().unwrap() {
                std::thread::sleep(Duration::from_millis(TIMESTEP));
                elapsed += TIMESTEP;

                if let Some(t) = thr_stime {
                    if elapsed >= t as u64 {
                        break;
                    }
                }
            }

            // Send stop request to tree
            thr_tree_tx
                .send(TreeReq::Done)
                .expect("failed to send stopreq to tree");

            // Join tree thread
            thr_tree_handle.join().expect("failed to join tree");

            // Join workers
            {
                let mut w_lock = thr_workers.lock().unwrap();

                while !w_lock.is_empty() {
                    w_lock.pop().unwrap().join();
                }
            }
        }));

        return true;
    }

    pub fn stop(&mut self) -> bool {
        if !self.handle.is_some() {
            return false;
        }

        println!("Stopping search.");

        *self.status_string.write().unwrap() = "stopping".to_string();
        *self.stopflag.write().unwrap() = true;
        self.handle
            .take()
            .unwrap()
            .join()
            .expect("failed joining search thread");
        *self.stopflag.write().unwrap() = false;
        *self.status_string.write().unwrap() = "idle".to_string();

        println!("Stopped.");
        return true;
    }

    pub fn status(&self) -> Status {
        Status {
            status: self.status_string.read().unwrap().clone(),
            workers: self
                .workers
                .lock()
                .unwrap()
                .iter()
                .map(|w| w.get_status())
                .collect(),
        }
    }
}

use crate::listener::Listener;
use crate::model::{Model, ModelPtr};
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
    handle: Option<JoinHandle<Tree>>,
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
        model: ModelPtr,
        thr_pos: Position,
        clients: Arc<Mutex<Listener>>,
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

        println!(
            "Searching `{}` for {}ms.",
            thr_pos.get_fen(),
            thr_stime.unwrap_or(0),
        );

        self.handle = Some(spawn(move || {
            let (thr_tree_tx, thr_tree_handle) = Tree::new(thr_pos).run();

            for _ in 0..num_cpus::get() {
                let mut new_worker =
                    Worker::new(thr_stopflag.clone(), thr_tree_tx.clone(), thr_model.clone());
                new_worker.start();
                thr_workers.lock().unwrap().push(new_worker);
            }

            *thr_status_string.write().unwrap() = "searching".to_string();

            const TIMESTEP: u64 = 100;
            let mut elapsed: u64 = 0;

            while !*thr_stopflag.read().unwrap() {
                std::thread::sleep(Duration::from_millis(TIMESTEP));
                elapsed += TIMESTEP;

                // Send status to clients
                clients.lock().unwrap().broadcast(&serde_json::to_string(&Status {
                    status: thr_status_string.read().unwrap().clone(),
                    workers: thr_workers
                        .lock()
                        .unwrap()
                        .iter()
                        .map(|w| w.get_status())
                        .collect(),
                }).expect("serialize failed").as_bytes());

                if let Some(t) = thr_stime {
                    if elapsed >= t as u64 {
                        break;
                    }
                }
            }

            // Send stop request to tree
            println!("Sending prestop to tree.");

            thr_tree_tx
                .send(TreeReq::RequestStop)
                .expect("failed to send stopreq to tree");

            println!("Waiting for workers.");
            // Join workers
            {
                let mut w_lock = thr_workers.lock().unwrap();

                while !w_lock.is_empty() {
                    w_lock.pop().unwrap().join();
                }
            }

            thr_tree_tx
                .send(TreeReq::Done)
                .expect("failed to send stopreq to tree");
            println!("Waiting for tree.");

            // Join tree thread
            let tree_ret = thr_tree_handle.join().expect("failed to join tree");

            return tree_ret;
        }));

        return true;
    }

    pub fn stop(&mut self) -> Option<Tree> {
        if !self.handle.is_some() {
            return None;
        }

        println!("Stopping search.");
        *self.stopflag.write().unwrap() = true;

        let ret = self.wait();
        
        *self.stopflag.write().unwrap() = false;
        println!("Stopped.");
        return ret;
    }

    pub fn wait(&mut self) -> Option<Tree> {
        if !self.handle.is_some() {
            return None;
        }

        *self.status_string.write().unwrap() = "stopping".to_string();

        let ret = self.handle
            .take()
            .unwrap()
            .join()
            .expect("failed joining search thread");

        *self.status_string.write().unwrap() = "idle".to_string();
        return Some(ret);
    }
}

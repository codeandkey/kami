use crate::net::Model;
use crate::position::Position;
use crate::tree::Tree;
use crate::worker::Worker;

use config::Config;
use std::error::Error;
use std::path::Path;
use tensorflow::SessionOptions;

use std::sync::{
    Arc, RwLock,
    atomic::{
        AtomicBool,
        Ordering
    }
};

pub struct Search {
    model: Arc<Model>,
    tree: Tree,
    state: RwLock<String>,
    workers: Vec<Worker>,
    stopflag: Arc<AtomicBool>,
    rootpos: Position,
}

impl Search {
    pub fn new(config: &Config) -> Result<Search, Box<dyn Error>> {
        let model_path = Path::new(&config.get_str("data_dir").unwrap()).join("model");

        Ok(Search {
            model: Arc::new(Model::load(&model_path, SessionOptions::new())?),
            tree: Tree::new(),
            state: RwLock::new("idle".to_string()),
            workers: Vec::new(),
            rootpos: Position::new(),
            stopflag: Arc::new(AtomicBool::from(false)),
        })
    }

    pub fn load(&mut self, p: Position) {
        self.stop();
        self.rootpos = p;
    }

    pub fn stop(&mut self) {
        if self.workers.is_empty() {
            return;
        }

        self.stopflag.store(true, Ordering::Relaxed);

        for worker in self.workers {
            worker.join();
        }

        self.stopflag.store(false, Ordering::Relaxed);
    }

    pub fn start(&mut self, num_workers: usize) {
        // Launch worker threads
        self.workers = (0..num_workers).map(|_| Worker::new(self.tree.clone(), self.stopflag.clone(), 16, self.rootpos.clone(), self.model.clone())).collect();
    }

}
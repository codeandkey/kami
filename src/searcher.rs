use crate::constants;
use crate::model::Model;
use crate::position::Position;
use crate::tree::{self, StatusResponse, Tree, TreeReq};
use crate::worker::{self, Worker};

use serde::Serialize;
use std::error::Error;
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
use std::thread::spawn;
use std::time::Duration;

/// Used
#[derive(Serialize, Clone)]
pub enum SearchStatus {
    Searching(Status),
    Done,
}

#[derive(Serialize, Clone)]
pub struct Status {
    pub tree: Option<tree::Status>,
    pub workers: Vec<worker::Status>,
    pub elapsed_ms: u64,
    pub rootfen: String,
    pub total_nodes: usize,
    pub total_batches: usize,
    pub maxnodes: Option<usize>,
    pub maxtime: Option<usize>,
    pub nps: f32,
    pub bps: f32,
}

impl Status {
    pub fn new() -> Self {
        Status {
            tree: None,
            workers: Vec::new(),
            elapsed_ms: 0,
            rootfen: "(no position)".to_string(),
            total_nodes: 0,
            total_batches: 0,
            maxnodes: None,
            maxtime: None,
            nps: 0.0,
            bps: 0.0,
        }
    }
}

pub struct Searcher {
    workers: Arc<Mutex<Vec<Worker>>>,
    position: Position,
    maxnodes: Option<usize>,
    maxtime: Option<usize>,
    model: Option<Arc<Model>>,
}

impl Searcher {
    pub fn new() -> Self {
        Searcher {
            workers: Arc::new(Mutex::new(Vec::new())),
            model: None,
            position: Position::new(),
            maxnodes: Some(constants::SEARCH_MAXNODES),
            maxtime: None,
        }
    }

    pub fn model(mut self, model: Arc<Model>) -> Self {
        self.model = Some(model);
        self
    }

    pub fn position(mut self, position: Position) -> Self {
        self.position = position;
        self
    }

    pub fn maxnodes(mut self, maxnodes: Option<usize>) -> Self {
        self.maxnodes = maxnodes;
        self
    }

    pub fn maxtime(mut self, maxtime: Option<usize>) -> Self {
        self.maxtime = maxtime;
        self
    }

    pub fn run<F, R>(self, f: F) -> Result<Tree, Box<dyn Error>>
    where
        F: Fn(Status) -> R,
    {
        let thr_model = self.model.as_ref().unwrap().clone();
        let thr_workers = self.workers.clone();
        let thr_rootfen = self.position.get_fen();
        let thr_position = self.position.clone();
        let thr_maxtime = self.maxtime.clone();
        let thr_maxnodes = self.maxnodes.clone();

        let (status_tx, status_rx) = channel();

        let handle = spawn(move || {
            let (thr_tree_tx, thr_tree_handle) =
                Tree::new(thr_position, constants::TEMPERATURE, constants::SEARCH_BATCH_SIZE).run();

            for _ in 0..num_cpus::get() {
                let mut new_worker = Worker::new(thr_tree_tx.clone(), thr_model.clone());
                new_worker.start();
                thr_workers.lock().unwrap().push(new_worker);
            }

            let mut elapsed: u64;
            let mut start_point = std::time::SystemTime::now();

            loop {
                std::thread::sleep(Duration::from_millis(constants::SEARCH_STATUS_RATE));
                elapsed = start_point.elapsed().unwrap().as_millis() as u64;

                // Get tree status
                let (tree_status_tx, tree_status_rx) = channel();
                thr_tree_tx
                    .send(TreeReq::GetStatus(tree_status_tx))
                    .expect("failed sending status req to tree");

                let tree_status = match tree_status_rx
                    .recv()
                    .expect("failed receiving status from tree")
                {
                    StatusResponse::NextStatus(s) => s,
                    StatusResponse::Stop => {
                        println!("Unexpected tree stop received waiting for status");
                        break;
                    }
                };

                // Reset starttime if no nodes have been searched yet (for GPU warmup)
                if tree_status.is_none() {
                    start_point = std::time::SystemTime::now();
                }

                let worker_status: Vec<worker::Status> = thr_workers
                    .lock()
                    .unwrap()
                    .iter()
                    .map(|w| w.get_status())
                    .collect();

                let (total_nodes, total_batches) = worker_status
                    .iter()
                    .map(|w| (w.total_nodes, w.batch_sizes.len()))
                    .fold((0, 0), |(a1, a2), (n, b)| (a1 + n, a2 + b));

                // Send status to clients
                status_tx
                    .send(SearchStatus::Searching(Status {
                        elapsed_ms: elapsed,
                        tree: tree_status,
                        rootfen: thr_rootfen.clone(),
                        workers: worker_status,
                        total_nodes: total_nodes,
                        total_batches: total_batches,
                        maxnodes: thr_maxnodes,
                        maxtime: thr_maxtime,
                        nps: (total_nodes as f32 / (elapsed + 1) as f32) * 1000.0,
                        bps: (total_batches as f32 / (elapsed + 1) as f32) * 1000.0,
                    }))
                    .expect("search status tx failed");

                if let Some(maxtime) = thr_maxtime {
                    if elapsed >= maxtime as u64 {
                        break;
                    }
                }
                
                if let Some(maxnodes) = thr_maxnodes {
                    if total_nodes >= maxnodes {
                        break;
                    }
                }
            }

            // Send stop request to tree
            thr_tree_tx
                .send(TreeReq::RequestStop)
                .expect("failed to send stopreq to tree");

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

            // Join tree thread
            let tree_ret = thr_tree_handle.join().expect("failed to join tree");

            // Send search stop signal
            status_tx
                .send(SearchStatus::Done)
                .expect("search status tx failed");

            return tree_ret;
        });

        loop {
            match status_rx.recv().expect("search status rx failed") {
                SearchStatus::Searching(stat) => {
                    f(stat);
                }
                SearchStatus::Done => break,
            }
        }

        let ret = handle.join().expect("failed joining search thread");

        return Ok(ret);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use chess::ChessMove;
    use std::str::FromStr;

    /// Generates a new threadsafe mock model for testing.
    pub fn mock() -> Arc<Model> {
        Arc::new(Model::Mock)
    }

    /// Tests that a search can be initialized.
    #[test]
    fn search_can_initialize() {
        Searcher::new();
    }

    /// Tests that a search can be started and immediately stopped.
    #[test]
    fn search_can_run_short() {
        Searcher::new()
            .maxtime(Some(500))
            .model(mock())
            .run(|_| ())
            .expect("search failed");
    }

    /// Tests that a short search will select a mate in 1 (with low temperature)
    #[test]
    fn search_sees_mate_in_one() {
        let mut pos = Position::new();

        pos.make_move(ChessMove::from_str("e2e4").expect("move parse fail"));
        pos.make_move(ChessMove::from_str("f7f6").expect("move parse fail"));
        pos.make_move(ChessMove::from_str("a2a4").expect("move parse fail"));
        pos.make_move(ChessMove::from_str("g7g5").expect("move parse fail"));

        let tree = Searcher::new()
            .position(pos)
            .model(mock())
            .maxnodes(Some(10000))
            .run(|_| ())
            .expect("search failed");

        assert_eq!(
            tree.select(),
            ChessMove::from_str("d1h5").expect("move parse fail"),
            "tree: \n{}",
            serde_json::to_string_pretty(&tree.get_status().unwrap()).expect("serialize failed"),
        );
    }

    /// Tests that a searcher status can serialize.
    #[test]
    fn search_status_can_serialize() {
        Searcher::new()
            .maxnodes(Some(50))
            .model(mock())
            .run(|stat| serde_json::to_string_pretty(&stat).expect("serialize failed"))
            .expect("search failed");
    }
}

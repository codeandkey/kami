use crate::model::ModelPtr;
use crate::position::Position;
use crate::tree::{self, StatusResponse, Tree, TreeReq};
use crate::worker::{self, Worker};

use serde::Serialize;
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Mutex};
use std::thread::{spawn, JoinHandle};
use std::time::Duration;

/// Used
#[derive(Serialize)]
pub enum SearchStatus {
    Searching(Status),
    Stopping,
    Done,
}

#[derive(Serialize)]
pub struct Status {
    tree: Option<tree::Status>,
    workers: Vec<worker::Status>,
    elapsed_ms: u64,
    rootfen: String,
}

impl Status {
    pub fn print(&self) {
        let total_nodes: usize = self.workers.iter().map(|x| x.total_nodes).sum();
        let total_batches: usize = self.workers.iter().map(|x| x.batch_sizes.len()).sum();

        println!("==> ################################################################### <==");

        println!(
            "==> Searching {}, {}ms elapsed, {} nodes ({:.1} nps), {} batches ({:.1} bps)",
            self.rootfen,
            self.elapsed_ms,
            total_nodes,
            total_nodes as f32 * 1000.0 / (self.elapsed_ms as f32),
            total_batches,
            total_batches as f32 * 1000.0 / (self.elapsed_ms as f32),
        );

        for (id, w) in self.workers.iter().enumerate() {
            println!(
                "=> Worker {:>2}: {:>16} | {:>7} nodes | {:>5} batches | {:.1} avg bsize",
                id,
                w.state,
                w.total_nodes,
                w.batch_sizes.len(),
                w.batch_sizes.iter().sum::<usize>() as f32 / w.batch_sizes.len() as f32,
            );
        }

        if let Some(tstatus) = &self.tree {
            tstatus.print();
        } else {
            println!("==> (No tree status available yet)");
        }

        println!("==> End search status");
    }
}

pub struct Searcher {
    workers: Arc<Mutex<Vec<Worker>>>,
    handle: Option<JoinHandle<Tree>>,
}

impl Searcher {
    pub fn new() -> Self {
        Searcher {
            workers: Arc::new(Mutex::new(Vec::new())),
            handle: None,
        }
    }

    pub fn start(
        &mut self,
        thr_stime: Option<usize>,
        model: ModelPtr,
        thr_pos: Position,
        temp: f32,
        batch_size: usize,
    ) -> Option<Receiver<SearchStatus>> {
        if self.handle.is_some() {
            println!("Search is already running!");
            return None;
        }

        let (search_status_tx, search_status_rx) = channel();

        let thr_model = model.clone();
        let thr_workers = self.workers.clone();
        let thr_rootfen = thr_pos.get_fen();

        println!(
            "Searching `{}` for {}ms.",
            thr_pos.get_fen(),
            thr_stime.unwrap_or(0),
        );

        self.handle = Some(spawn(move || {
            let (thr_tree_tx, thr_tree_handle) = Tree::new(thr_pos, temp, batch_size).run();

            for _ in 0..num_cpus::get() {
                let mut new_worker = Worker::new(thr_tree_tx.clone(), thr_model.clone());
                new_worker.start();
                thr_workers.lock().unwrap().push(new_worker);
            }

            const TIMESTEP: u64 = 100;
            let mut elapsed: u64;
            let start_point = std::time::SystemTime::now();

            loop {
                std::thread::sleep(Duration::from_millis(TIMESTEP));
                elapsed = start_point.elapsed().unwrap().as_millis() as u64;

                // Get tree status
                let (status_tx, status_rx) = channel();
                thr_tree_tx
                    .send(TreeReq::GetStatus(status_tx))
                    .expect("failed sending status req to tree");

                let tree_status = match status_rx.recv().expect("failed receiving status from tree")
                {
                    StatusResponse::NextStatus(s) => s,
                    StatusResponse::Stop => {
                        println!("Unexpected tree stop received waiting for status");
                        break;
                    }
                };

                // Send status to clients
                let _ = search_status_tx.send(SearchStatus::Searching(Status {
                    elapsed_ms: elapsed,
                    tree: tree_status,
                    rootfen: thr_rootfen.clone(),
                    workers: thr_workers
                        .lock()
                        .unwrap()
                        .iter()
                        .map(|w| w.get_status())
                        .collect(),
                }));

                if let Some(t) = thr_stime {
                    if elapsed >= t as u64 {
                        break;
                    }
                }
            }

            // Send search stopping signal
            let _ = search_status_tx.send(SearchStatus::Stopping);

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

            // Send search stop signal
            let _ = search_status_tx.send(SearchStatus::Done);

            return tree_ret;
        }));

        return Some(search_status_rx);
    }

    pub fn wait(&mut self) -> Option<Tree> {
        if !self.handle.is_some() {
            return None;
        }

        println!("Waiting for search thread to join..");

        let ret = self
            .handle
            .take()
            .unwrap()
            .join()
            .expect("failed joining search thread");

        println!("Search thread joined!");

        return Some(ret);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::model::{self, mock::MockModel, Model};
    use chess::ChessMove;
    use std::str::FromStr;

    /// Tests that a search can be initialized.
    #[test]
    fn search_can_initialize() {
        Searcher::new();
    }

    /// Tests that a search can be started and immediately stopped.
    #[test]
    fn search_can_run_short() {
        let mut search = Searcher::new();
        let rx = search
            .start(
                Some(500),
                model::make_ptr(MockModel::generate().expect("model gen failed")),
                Position::new(),
                1.0,
                4,
            )
            .unwrap();

        loop {
            match rx.recv().expect("rx failed") {
                SearchStatus::Done => break,
                SearchStatus::Searching(stat) => stat.print(),
                _ => (),
            }
        }

        search.wait();
    }

    /// Tests that a short search will select a mate in 1 (with low temperature)
    #[test]
    fn search_sees_mate_in_one() {
        let mut pos = Position::new();

        pos.make_move(ChessMove::from_str("e2e4").expect("move parse fail"));
        pos.make_move(ChessMove::from_str("f7f6").expect("move parse fail"));
        pos.make_move(ChessMove::from_str("a2a4").expect("move parse fail"));
        pos.make_move(ChessMove::from_str("g7g5").expect("move parse fail"));

        let mut search = Searcher::new();
        let rx = search
            .start(Some(500), model::make_ptr(MockModel::generate().expect("model gen failed")), pos, 0.1, 4)
            .unwrap();

        loop {
            match rx.recv().expect("rx failed") {
                SearchStatus::Done => break,
                SearchStatus::Searching(stat) => stat.print(),
                _ => (),
            }
        }

        let final_tree = search.wait().expect("no tree returned");

        assert_eq!(
            final_tree.select(),
            ChessMove::from_str("d1h5").expect("move parse fail")
        );
    }

    /// Tests that a stopped search cannot stop again.
    #[test]
    fn search_stop_already_stopped() {
        let mut search = Searcher::new();
        assert!(search.wait().is_none());
    }

    /// Tests that a started search cannot start again.
    #[test]
    fn search_start_already_started() {
        let mut search = Searcher::new();
        let pos = Position::new();

        search
            .start(
                Some(200),
                model::make_ptr(MockModel::generate().expect("model gen failed")),
                pos.clone(),
                1.0,
                4,
            )
            .unwrap();

        assert!(search
            .start(Some(200), model::make_ptr(MockModel::generate().expect("model gen failed")), pos, 1.0, 4)
            .is_none());

        search.wait().expect("no tree returned");
    }

    /// Tests that a searcher status can serialize.
    #[test]
    fn search_status_can_serialize() {
        let pos = Position::new();

        let mut search = Searcher::new();
        let rx = search
            .start(Some(200), model::make_ptr(MockModel::generate().expect("model gen failed")), pos, 1.0, 4)
            .unwrap();

        loop {
            match rx.recv().expect("rx failed") {
                SearchStatus::Done => break,
                SearchStatus::Searching(stat) => print!(
                    "{}",
                    serde_json::to_string_pretty(&stat).expect("serialize failed")
                ),
                _ => (),
            }
        }

        search.wait().expect("no tree returned");
    }
}

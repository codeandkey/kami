use crate::net::Model;
use chess::Board;
use config::Config;
use std::error::Error;
use std::path::Path;
use std::sync::RwLock;
use std::sync::atomic::AtomicBool;
use std::thread::Thread;
use tensorflow::SessionOptions;

use serde::Serialize;

#[derive(Clone, Serialize)]
pub struct NodeValue {
    n: u32,
    w: f32,
}

impl NodeValue {
    pub fn new() -> Self {
        NodeValue {
            n: 0,
            w: 0.0,
        }
    }
}

#[derive(Serialize)]
pub struct TreeStatus {
    values: Vec<NodeValue>,
    actions: Vec<String>,
    p: Vec<f32>,
}

impl TreeStatus {
    pub fn add(&mut self, v: NodeValue, action: String, p: f32) {
        self.values.push(v);
        self.actions.push(action);
        self.p.push(p);
    }
}

pub struct Tree {
    children: Vec<Box<Tree>>,
    claim: AtomicBool,
    value: RwLock<NodeValue>,
    p: RwLock<Option<f32>>,
    action: String,
}

impl Tree {
    pub fn root() -> Self {
        Tree {
            children: Vec::new(),
            claim: AtomicBool::from(false),
            value: RwLock::new(NodeValue::new()),
            p: RwLock::new(None),
            action: "none".to_string(),
        }
    }

    pub fn status(&self) -> TreeStatus {
        TreeStatus {
            values: self.children.iter().map(|c| c.value.read().unwrap().clone()).collect(),
            actions: self.children.iter().map(|c| c.action.clone()).collect(),
            p: self.children.iter().map(|c| c.p.read().unwrap().unwrap_or(-1.0)).collect(),
        }
    }
}

#[derive(Serialize)]
pub struct Status {
    state: String,
    tree: TreeStatus,
}

pub struct Search {
    model: Model,
    tree: Tree,
    state: RwLock<String>,
    workers: Vec<Thread>,
}

impl Search {
    pub fn new(config: &Config) -> Result<Search, Box<dyn Error>> {
        let model_path = Path::new(&config.get_str("data_dir").unwrap()).join("model");

        Ok(Search {
            model: Model::load(&model_path, SessionOptions::new())?,
            tree: Tree::root(),
            state: RwLock::new("idle".to_string()),
            workers: Vec::new(),
        })
    }

    pub fn status(&self) -> Status {
        Status {
            state: self.state.read().unwrap().clone(),
            tree: self.tree.status(),
        }
    }

    pub fn advance(&mut self, b: Board, time: u32) -> String {
        "".to_string()
    }
}
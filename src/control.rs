use crate::net::Model;
use crate::position::Position;
use crate::searcher::{self, Searcher};
use crate::tree::{Tree, TreeReq};

use chess::ChessMove;
use config::Config;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::str::FromStr;
use std::thread::{spawn, JoinHandle};
use std::time::{Duration, SystemTime};

use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc::{channel, Receiver, Sender},
    Arc, RwLock,
};

#[derive(Serialize, Deserialize)]
pub struct Message {
    mtype: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    code: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    cmove: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    stime: Option<usize>,

    #[serde(skip_serializing_if = "Option::is_none")]
    search_status: Option<searcher::Status>,
}

impl Message {
    pub fn error(code: impl ToString) -> Message {
        Message {
            mtype: "error".to_string(),
            code: Some(code.to_string()),
            cmove: None,
            stime: None,
            search_status: None,
        }
    }

    pub fn simple(mtype: impl ToString) -> Message {
        Message {
            mtype: mtype.to_string(),
            code: None,
            cmove: None,
            stime: None,
            search_status: None,
        }
    }
}

pub struct Control {
    model: Arc<Model>,
    state: RwLock<String>,
    pos: Position,
    searcher: Searcher,
    queue: VecDeque<(String, Sender<String>)>,
}

impl Control {
    pub fn new(model: Arc<Model>, config: &Config) -> Control {
        Control {
            model: model,
            state: RwLock::new("idle".to_string()),
            queue: VecDeque::new(),
            pos: Position::new(),
            searcher: Searcher::new(),
        }
    }

    pub fn execute(&mut self, json: String) -> Receiver<String> {
        let (tx, rx) = channel();
        self.queue.push_back((json, tx));
        rx
    }

    pub fn process(&mut self) {
        while !self.queue.is_empty() {
            // Pop next command from queue
            let (content, tx) = self.queue.pop_front().unwrap();

            // Deserialize into command
            let cmd = match serde_json::from_str::<Message>(&content) {
                Ok(c) => c,
                Err(e) => {
                    tx.send(serde_json::to_string(&Message::error(e)).unwrap())
                        .expect("tx failed");
                    continue;
                }
            };

            // execute the command and return the response
            tx.send(serde_json::to_string(&self.do_command(cmd)).unwrap())
                .expect("tx failed")
        }
    }

    pub fn do_command(&mut self, c: Message) -> Message {
        if c.mtype == "search" {
            return self.search(c);
        }

        if c.mtype == "make_move" {
            return self.make_move(c);
        }

        if c.mtype == "reset" {
            return self.reset();
        }

        if c.mtype == "stop" {
            return self.stop(c);
        }

        if c.mtype == "status" {
            return self.status();
        }

        Message::error(format!("Unknown command '{}'", c.mtype))
    }

    fn search(&mut self, c: Message) -> Message {
        if self
            .searcher
            .start(c.stime, self.model.clone(), self.pos.clone())
        {
            Message::simple("ok")
        } else {
            Message::error("Search is already running!")
        }
    }

    fn stop(&mut self, c: Message) -> Message {
        if self.searcher.stop() {
            Message::simple("ok")
        } else {
            Message::error("Search is not running!")
        }
    }

    fn status(&self) -> Message {
        Message {
            mtype: "status".to_string(),
            search_status: Some(self.searcher.status()),
            cmove: None,
            stime: None,
            code: None,
        }
    }

    fn make_move(&mut self, c: Message) -> Message {
        if c.cmove.is_none() {
            return Message::error("Missing cmove");
        }

        if let Ok(mv) = ChessMove::from_str(&c.cmove.unwrap()) {
            if !self.pos.make_move(mv) {
                Message::error("Invalid move")
            } else {
                Message::simple(self.pos.get_fen())
            }
        } else {
            Message::error("Invalid cmove")
        }
    }

    fn reset(&mut self) -> Message {
        self.pos = Position::new();
        Message::simple("ok")
    }
}

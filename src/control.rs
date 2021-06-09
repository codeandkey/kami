use crate::model::{Model, ModelPtr};
use crate::position::Position;
use crate::searcher::{self, Searcher};

use chess::ChessMove;
use config::Config;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::str::FromStr;

use std::sync::{
    mpsc::{channel, Receiver, Sender}, Arc
};

/// Incoming or outgoing message to the control interface.
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

/// Manages the state of the application. Recieves control messages in JSON format and responds to each message through mpsc channels.
pub struct Control {
    model: ModelPtr,
    pos: Position,
    searcher: Searcher,
    queue: VecDeque<(String, Sender<String>)>,
}

impl Control {
    /// Returns a new control instance.
    pub fn new(model: ModelPtr, _: &Config) -> Control {
        Control {
            model: model,
            queue: VecDeque::new(),
            pos: Position::new(),
            searcher: Searcher::new(),
        }
    }

    /// Adds a command to the queue. 
    /// Returns a channel which the response will be sent to.
    pub fn execute(&mut self, json: String) -> Receiver<String> {
        let (tx, rx) = channel();
        self.queue.push_back((json, tx));
        rx
    }

    /// Processes all pending commands.
    /// Returns true if the application should exit, or false if more commands may be received.
    pub fn process(&mut self) -> bool {
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
            let (resp, should_quit) = self.do_command(cmd);

            tx.send(serde_json::to_string(&resp).unwrap())
                .expect("tx failed");

            if should_quit {
                return true;
            }
        }

        return false;
    }

    /// Executes a single command in Message format.
    /// Returns the reponse along with a flag indicating whether the program should exit.
    fn do_command(&mut self, c: Message) -> (Message, bool) {
        if c.mtype == "search" {
            return (self.search(c), false);
        }

        if c.mtype == "make_move" {
            return (self.make_move(c), false);
        }

        if c.mtype == "reset" {
            return (self.reset(), false);
        }

        if c.mtype == "stop" {
            return (self.stop(), false);
        }

        if c.mtype == "status" {
            return (self.status(), false);
        }

        if c.mtype == "quit" {
            return (Message::simple("ok"), true);
        }

        (Message::error(format!("Unknown command '{}'", c.mtype)), false)
    }

    /// Handles a search command.
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

    /// Handles a stop command.
    fn stop(&mut self) -> Message {
        if self.searcher.stop() {
            Message::simple("ok")
        } else {
            Message::error("Search is not running!")
        }
    }

    /// Handles a status command.
    fn status(&self) -> Message {
        Message {
            mtype: "status".to_string(),
            search_status: Some(self.searcher.status()),
            cmove: None,
            stime: None,
            code: None,
        }
    }

    /// Handles a make_move command.
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

    /// Handles a reset command.
    fn reset(&mut self) -> Message {
        self.pos = Position::new();
        Message::simple("ok")
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// Tests the control can be initialized.
    #[test]
    fn control_can_initialize() {
        
    }
}
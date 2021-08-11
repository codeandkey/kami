/// Kami searcher entry point.

mod batch;
mod consts;
mod model;
mod node;
mod params;
mod position;
mod tree;
mod worker;

use model::Model;
use position::Position;
use tree::Tree;
use worker::{Worker, WorkerMsg};

use chess::ChessMove;
use serde::{Serialize, Deserialize};
use std::error::Error;
use std::io::{BufReader, BufRead, Write, BufWriter};
use std::net::TcpListener;
use std::str::FromStr;
use std::sync::{Arc, mpsc::channel};

#[derive(Serialize)]
/// Outgoing message type.
enum Message {
    Error(String),
    Searching(Tree),
    Done(Tree),
}

#[derive(Serialize, Deserialize)]
/// Incoming message type from control client.
enum Command {
    Push(String),
    Go,
    Load(Vec<String>),
    Config(params::Params),
    Stop,
}

/// Parses the next command from a client stream.
fn next_command(reader: &mut impl BufRead) -> Result<Command, Box<dyn Error>> {
    let mut msg = String::new();
    reader.read_line(&mut msg)?;
    Ok(serde_json::from_str::<Command>(&msg)?)
}

/// Writes an error message to a client.
fn write_error(client: &mut impl Write, message: impl ToString) -> Result<(), Box<dyn Error>> {
    println!("Error: {}", message.to_string());
    write_message(client, Message::Error(message.to_string()))
}

/// Writes a generic message to a client.
fn write_message(client: &mut impl Write, message: Message) -> Result<(), Box<dyn Error>> {
    client.write(serde_json::to_string(&message)?.as_bytes())?;
    client.write(b"\n")?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // Start listening for client
    let bind_addr = format!("0.0.0.0:{}", consts::PORT);
    let socket = TcpListener::bind(&bind_addr)?;
    
    println!("Waiting for client on {}", bind_addr);
    
    let (client, addr) = socket.accept().expect("socket accept failed");

    println!("Accepted connection from {}", addr);

    let mut reader = BufReader::new(&client);
    let mut writer = BufWriter::new(&client);

    // Wait for initial configuration
    let mut config = match next_command(&mut reader)? {
        Command::Config(p) => p,
        _ => {
            return write_error(&mut writer, "Invalid first message, expected configuration");
        }
    };

    println!("Loaded initial configuration:");
    println!("{}", serde_json::to_string_pretty(&config)?);

    let mut tree = Tree::new(Position::new(), &config);
    let model = Arc::new(Model::load(&config.model_path)?);

    let (wtx, wrx) = channel();

    // Spawn workers.
    let workers: Vec<Worker> = (0..config.num_threads)
        .map(|_| Worker::new(wtx.clone(), model.clone()))
        .collect();

    loop {
        let next = next_command(&mut reader)?;
        match next {
            Command::Stop => break, // Stop server.
            Command::Config(p) => config = p, // Load new params.
            Command::Load(actions) => {
                // Load new tree.
                let mut new_pos = Position::new();

                for a in actions {
                    assert!(new_pos.make_move(ChessMove::from_str(&a).expect("invalid move")));
                }

                tree = Tree::new(new_pos, &config);
            },
            Command::Push(action) => {
                // Advance tree.
                tree.push(ChessMove::from_str(&action).expect("invalid move"));
            }
            Command::Go => {
                // Start searching!
                while tree[0].n < config.search_nodes as u32 {
                    match wrx.recv().unwrap() {
                        WorkerMsg::Ready(tx) => {
                            tx.send(Some(Box::new(tree.next_batch()))).unwrap();
                        },
                        WorkerMsg::Expand(res) => {
                            tree.expand(*res);
                            write_message(&mut writer, Message::Searching(tree.clone()))?;
                        },
                    }
                }

                // Ensure we've finished all the expansions in the channel
                let mut readies = Vec::new();

                while let Ok(next) = wrx.try_recv() {
                    match next {
                        WorkerMsg::Expand(res) => {
                            tree.expand(*res);
                        },
                        WorkerMsg::Ready(tx) => {
                            readies.push(WorkerMsg::Ready(tx));
                        },
                    }
                }

                // Re-add the ready messages to the channel so they are picked up
                // in the next search.

                readies.into_iter().for_each(|x| wtx.send(x).unwrap());

                // Send a final search-complete message.
                write_message(&mut writer, Message::Done(tree.clone()))?;
            },
            x => println!("{}", serde_json::to_string_pretty(&x)?),
        }
    }

    workers.into_iter().for_each(|x| x.join());

    Ok(())
}
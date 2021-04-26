use crate::net::Model;
use serde_json::Value;
use std::collections::HashMap;
use std::error::Error;
use std::io;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{Sender, Receiver, channel};
use std::net::{TcpListener, SocketAddr, TcpStream};
use std::io::{BufReader, BufWriter, BufRead, Read, Write};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use tensorflow::SessionOptions;

pub struct Server {
    stop: Sender<()>,
    dispatch: Option<JoinHandle<thread::Result<()>>>,
}

impl Server {
    pub fn start(data_dir: &Path) -> Result<Server, Box<dyn Error>> {
        let model = Arc::new(Model::load(&data_dir.join("model"), SessionOptions::new())?);
        let srv = TcpListener::bind("127.0.0.1:2191")?;
        let (tx, rx) = std::sync::mpsc::channel::<()>();

        srv.set_nonblocking(true)?;

        let dispatch_model = model.clone();
        let dispatch = thread::spawn(|| Server::dispatch(srv, rx, dispatch_model));

        Ok(Server {
            dispatch: Some(dispatch),
            stop: tx,
        })
    }

    pub fn stop(&mut self) {
        info!("Stopping server.");
        self.stop.send(());

        match self.dispatch.take().unwrap().join() {
            Ok(_) => {},
            Err(e) => {
                warn!("Dispatch thread exited with error: {:?}", e);
            }
        }

        info!("Stopped server.");
    }

    fn dispatch(srv: TcpListener, stop: Receiver<()>, model: Arc<Model>) -> thread::Result<()> {
        let mut client_threads: Vec<thread::JoinHandle<thread::Result<()>>> = Vec::new();
        let mut client_stops: Vec<Sender<()>> = Vec::new();

        info!("Listening on {}", srv.local_addr().unwrap());
        info!("Ready for connections.");

        for stream in srv.incoming() {
            match stop.try_recv() {
                Ok(_) => {
                    debug!("Received stop signal");
                    break;
                },
                Err(e) => {
                    if e != std::sync::mpsc::TryRecvError::Empty {
                        return Err(Box::new(e));
                    }
                }
            }

            let client: TcpStream = match stream {
                Ok(s) => s,
                Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                    debug!("Ignoring EWOULDBLOCK");
                    thread::sleep(Duration::from_millis(250));
                    continue;
                },
                Err(e) => panic!("I/O error: {}", e),
            };

            let (tx, rx) = std::sync::mpsc::channel::<()>();

            
            client_threads.push(thread::spawn(move || Server::connection_handler(client, rx)));
            client_stops.push(tx);
        }

        for stop in client_stops {
            stop.send(());
        }

        for client in client_threads {
            match client.join() {
                Ok(_) => {},
                Err(e) => {
                    warn!("Client thread exited with error: {:?}", e);
                }
            }
        }

        Ok(())
    }

    fn connection_handler(s: TcpStream, stop: Receiver<()>) -> thread::Result<()> {
        let rx = BufReader::new(&s);
        let mut tx = BufWriter::new(&s);

        info!("Accepted connection from {}", s.peer_addr().unwrap());

        for line in rx.lines() {
            let line = line.expect("socket read failed");

            let value: Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(e) => {
                    warn!("Error parsing JSON: {}", e);
                    continue;
                },
            };

            let resp = Server::message_handler(value);
            tx.write(serde_json::to_string(&resp).expect("serialize failed").as_bytes()).expect("tx failed");

            match stop.try_recv() {
                Ok(_) => {
                    debug!("{}: Received stop signal", s.peer_addr().unwrap());
                    break;
                },
                Err(e) => {
                    if e != std::sync::mpsc::TryRecvError::Empty {
                        return Err(Box::new(e));
                    }
                }
            }
        }

        info!("Dropping peer {}", s.peer_addr().unwrap());
        Ok(())
    }

    fn message_handler(value: serde_json::Value) -> serde_json::Value {
        let mut resp: HashMap<String, Value> = HashMap::new();

        match value.get("id") {
            Some(i) => {
                resp.insert("id".to_string(), i.clone());
            },
            None => (),
        };

        let mtype = value.get("type");
        
        if mtype.is_none() {
            resp.insert("type".to_string(), Value::String("error".to_string()));
            resp.insert("message".to_string(), Value::String("Missing message type.".to_string()));
            return serde_json::to_value(resp).expect("JSON conversion failed");
        }

        let mtype = mtype.unwrap();

        if mtype == "ping" {
            resp.insert("type".to_string(), Value::String("pong".to_string()));
            resp.insert("content".to_string(), Value::String("Pong!".to_string()));
        }

        if mtype == "stop" {
            resp.insert("type".to_string(), Value::String("stopping".to_string()));
            info!("Compute server received stop request.");
        }

        return serde_json::to_value(resp).expect("JSON conversion failed");
    }
}
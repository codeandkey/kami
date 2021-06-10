use std::error::Error;
use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread::spawn;

pub struct Listener {
    port: u16,
    clients: Arc<Mutex<Vec<TcpStream>>>,
}

impl Listener {
    pub fn new(port: u16) -> Self {
        Listener {
            port: port,
            clients: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn start(&mut self) -> Result<(), Box<dyn Error>> {
        let bind_addr = format!("0.0.0.0:{}", self.port);
        let listener = TcpListener::bind(&bind_addr)?;
        let thr_clients = self.clients.clone();

        println!("Listening on {}", bind_addr);

        spawn(move || {
            for stream in listener.incoming() {
                match stream {
                    Ok(stream) => {
                        println!("{}: accepted connection", stream.peer_addr().unwrap());

                        thr_clients.lock().unwrap().push(stream);
                    }
                    Err(e) => {
                        println!("Accept failed: {}", e);
                    }
                }
            }
        });

        Ok(())
    }

    pub fn broadcast(&mut self, message: &[u8]) {
        self.clients.lock().unwrap().retain(|mut c| {
            if let Err(e) = c.write(message) {
                println!("{}: dropping connection [{}]", c.peer_addr().unwrap(), e);
                return false;
            } else {
                return true;
            }
        });
    }
}

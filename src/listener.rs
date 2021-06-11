use std::error::Error;
use std::io::{self, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::thread::{self, spawn, JoinHandle};

pub struct Listener {
    port: u16,
    clients: Arc<Mutex<Vec<TcpStream>>>,
    stopflag: Arc<Mutex<bool>>,
    handle: Option<JoinHandle<()>>,
}

impl Listener {
    pub fn new(port: u16) -> Self {
        Listener {
            port: port,
            clients: Arc::new(Mutex::new(Vec::new())),
            stopflag: Arc::new(Mutex::new(false)),
            handle: None,
        }
    }

    pub fn start(mut self) -> Result<Self, Box<dyn Error>> {
        let bind_addr = format!("0.0.0.0:{}", self.port);

        let listener = TcpListener::bind(&bind_addr)?;
        listener.set_nonblocking(true).expect("failed to set nonblocking mode");

        println!("Listening on {}", bind_addr);

        let thr_clients = self.clients.clone();
        let thr_stopflag = self.stopflag.clone();

        self.handle = Some(spawn(move || {
            for stream in listener.incoming() {
                match stream {
                    Ok(stream) => {
                        println!("{}: accepted connection", stream.peer_addr().unwrap());

                        thr_clients.lock().unwrap().push(stream);
                    }
                    Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                        if *thr_stopflag.lock().unwrap() {
                            break;
                        } else {
                            thread::sleep(Duration::from_millis(100));
                            continue;
                        }
                    }
                    Err(e) => {
                        panic!("Accept failed: {}", e);
                    }
                }
            }

            thr_clients.lock().unwrap().clear();
            println!("Stopped listener.");
        }));

        Ok(self)
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

    pub fn stop(&mut self) {
        *self.stopflag.lock().unwrap() = true;
        self.handle.take().unwrap().join().expect("listener join failed");
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::io::Read;

    /// Tests the listener can be initialized.
    #[test]
    fn listener_can_initialize() {
        Listener::new(2961);
    }

    /// Tests the listener can be started and stopped.
    #[test]
    fn listener_can_run() {
        Listener::new(2961).start().expect("start failed").stop();
    }

    /// Tests the listener can broadcast commands to a mock client.
    #[test]
    fn listener_can_broadcast() {
        let mut listen = Listener::new(2961).start().expect("start failed");
        
        // Connect to listener with some clients
        let mut clients: Vec<TcpStream> = (0..4).map(|_| TcpStream::connect("127.0.0.1:2961").expect("connect failed")).collect();

        // Drop one client
        clients.pop();

        thread::sleep(Duration::from_millis(1000));

        // Broadcast a "hello"
        println!("Broadcasting hello");
        listen.broadcast(b"hello");

        // Receive a hello on all clients.
        clients.iter_mut().for_each(|c| {
            println!("Waiting for hello..");
            let mut buf = [0u8; 16];
            let len = c.read(&mut buf).expect("read failed");
            assert_eq!(&buf[..len], b"hello");
            println!("Verified");
        });

        clients.clear();
        listen.stop();
    }
}
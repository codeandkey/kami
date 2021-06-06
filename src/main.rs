extern crate clap;
extern crate dirs;
extern crate pretty_env_logger;
extern crate serde;
extern crate tch;

#[macro_use]
extern crate log;

mod batch;
mod control;
mod net;
mod node;
mod perft;
mod position;
mod searcher;
mod tree;
mod worker;

use clap::{App, Arg};
use config::Config;

use std::error::Error;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread::{spawn, JoinHandle};

/**
 * Server entry point
 */

fn main() -> Result<(), Box<dyn Error>> {
    // Parse arguments
    let matches = App::new("kami")
        .version(env!("CARGO_PKG_VERSION"))
        .author("Justin Stanley <jtst@iastate.edu>")
        .about("A chess engine powered by reinforcement learning.")
        .arg(
            Arg::with_name("verbose")
                .short("v")
                .long("verbose")
                .help("Enables verbose logging"),
        )
        .get_matches();

    // Initialize config
    let mut config = Config::new();

    config.set_default("verbose", false)?;
    config.set_default(
        "data_dir",
        dirs::data_dir().unwrap().join("kami").to_str().unwrap(),
    )?;

    // Initialize logging
    if matches.is_present("verbose") {
        config.set("verbose", true)?;
    }

    std::env::set_var("RUST_LOG", "warn");

    if config.get_bool("verbose")? {
        std::env::set_var("RUST_LOG", std::env::var("RUST_LOG").unwrap() + ",debug");
    }

    pretty_env_logger::init();

    // Initialize model
    let model = Arc::new(net::Model::load(
        Path::new(&config.get_str("data_dir").unwrap())
            .join("model")
            .as_path()
    )?);

    // Start control
    let control = Arc::new(Mutex::new(control::Control::new(model.clone(), &config)));

    // Wait for connections
    const BIND: &str = "0.0.0.0:2961";

    let listener = TcpListener::bind(BIND)?;
    let mut clients: Vec<JoinHandle<()>> = Vec::new();

    println!("Listening on {}", BIND);

    let listener_control = control.clone();
    let listener_thread = spawn(move || {
        for stream in listener.incoming() {
            match stream {
                Ok(mut stream) => {
                    println!("{}: accepted connection", stream.peer_addr().unwrap());

                    let client_control = listener_control.clone();
                    let addr = stream.peer_addr().unwrap();

                    clients.push(spawn(move || {
                        loop {
                            let mut buf = [0u8; 512];

                            if let Err(e) = stream.read(&mut buf) {
                                println!("{}: read fail: {}", addr, e);
                                break;
                            }

                            let trim_chars: &[_] = &['\n', '\r', ' ', '\t', '\0'];
                            let contents = String::from_utf8_lossy(&buf)
                                .trim_matches(trim_chars)
                                .to_string();

                            let rx = client_control.lock().unwrap().execute(contents);

                            if let Err(e) =
                                stream.write(rx.recv().expect("control recv failed").as_bytes())
                            {
                                println!("{}: write fail: {}", addr, e);
                                break;
                            }
                        }

                        println!("{}: dropping connection", addr);
                    }));
                }
                Err(e) => {
                    println!("Accept failed: {}", e);
                }
            }
        }
    });

    // Start processing commands
    println!("Starting command processing loop.");

    loop {
        control.lock().unwrap().process();
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    listener_thread
        .join()
        .expect("failed joining listener thread");
    Ok(())
}

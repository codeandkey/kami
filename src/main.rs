extern crate clap;
extern crate dirs;
#[macro_use] extern crate log;
extern crate pretty_env_logger;
extern crate tensorflow;

mod net;
mod server;

use clap::{Arg, App};
use std::error::Error;
use std::path::PathBuf;
use server::Server;


fn main() -> Result<(), Box<dyn Error>> {
    let matches = App::new("kami")
        .version(env!("CARGO_PKG_VERSION"))
        .author("Justin Stanley <jtst@iastate.edu>")
        .about("portable chess engine powered by reinforcement learning")
        .arg(Arg::with_name("verbose")
            .short("v")
            .long("verbose")
            .help("Enables verbose logging"))
        .get_matches();

    // Verbose logging option
    if matches.is_present("verbose") {
        std::env::set_var("RUST_LOG", "kami");
    }

    pretty_env_logger::init();

    info!("Verbose logging enabled.");

    // Set data dir.
    let data_dir = dirs::data_dir().unwrap().join("kami");

    if !PathBuf::from(&data_dir).exists() {
        panic!("Data directory \"{}\" does not exist! To create it:\n\tpython generate.py", data_dir.display());
    }

    info!("Found data dir \"{}\".", data_dir.display());

    // Start compute server.
    let mut srv = Server::start(&data_dir)?;

    // Run TUI
    // ...

    let mut line = String::new();
    std::io::stdin().read_line(&mut line).unwrap();

    info!("Shutting down.");
    srv.stop();

    info!("Bye!");
    Ok(())
}

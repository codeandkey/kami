extern crate clap;
extern crate dirs;
#[macro_use] extern crate log;
extern crate pretty_env_logger;
extern crate tensorflow;

use clap::{Arg, App};
use tensorflow::SessionOptions;

mod net;

fn main() {
    let matches = App::new("kami")
        .version(env!("CARGO_PKG_VERSION"))
        .author("Justin Stanley <jtst@iastate.edu>")
        .about("portable chess engine powered by reinforcement learning")
        .arg(Arg::with_name("verbose")
            .short("v")
            .long("verbose")
            .help("Enables verbose logging"))
        .get_matches();

    // Init data dirs
    let data_dir = dirs::data_dir().unwrap().join("kami");

    // Verbose logging option
    if matches.is_present("v") {
        std::env::set_var("RUST_LOG", "kami");
    }

    pretty_env_logger::init();

    info!("Verbose logging enabled.");
    info!("Using data dir \"{}\".", data_dir.display());

    // Setup tensorflow
    info!("Loading latest model..");

    net::Model::load(&data_dir.join("model"), SessionOptions::new()).expect("Failed to load model.");
}

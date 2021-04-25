extern crate clap;
extern crate dirs;
#[macro_use] extern crate log;
extern crate pretty_env_logger;
extern crate tensorflow;

use clap::{Arg, App, SubCommand};
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
        .arg(Arg::with_name("initdata")
            .long("initdata")
            .help("Initialize data directory"))
        .get_matches();

    // Init data dirs
    let data_dir = dirs::data_dir().unwrap().join("kami");

    if matches.is_present("initdata") {
        std::fs::create_dir_all(&data_dir).expect("Failed to initialize data directory");
        println!("{}", data_dir.display());
        return;
    }

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

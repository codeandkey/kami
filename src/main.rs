
#![feature(proc_macro_hygiene, decl_macro)]

extern crate clap;
extern crate dirs;
#[macro_use] extern crate log;
extern crate pretty_env_logger;
extern crate tensorflow;

#[macro_use]
extern crate rocket;

mod net;

use clap::{Arg, App};
use config::Config;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let matches = App::new("kami")
        .version(env!("CARGO_PKG_VERSION"))
        .author("Justin Stanley <jtst@iastate.edu>")
        .about("A portable chess engine powered by reinforcement learning.")
        .arg(Arg::with_name("verbose")
            .short("v")
            .long("verbose")
            .help("Enables verbose logging"))
        .get_matches();

    let mut config = Config::new();

    // Define config options
    config.set_default("verbose", false)?;
    config.set_default("data_dir", dirs::data_dir().unwrap().join("kami").to_str().unwrap())?;

    // Verbose logging option
    if matches.is_present("verbose") {
        config.set("verbose", true)?;
    }

    std::env::set_var("RUST_LOG", "warn");

    if config.get_bool("verbose")? {
        std::env::set_var("RUST_LOG", std::env::var("RUST_LOG").unwrap() + ",debug"); 
    }

    pretty_env_logger::init();

    Ok(())
}

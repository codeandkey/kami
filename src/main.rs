
extern crate clap;
extern crate dirs;
extern crate pretty_env_logger;
extern crate actix_web;
extern crate serde;
extern crate tensorflow;

#[macro_use] extern crate log;

mod input;
mod net;
mod perft;
mod position;
mod search;

static BIND: &'static str = "0.0.0.0:8000";

use clap::{Arg, App};
use config::Config;

use actix_web::{
    get, post, web, HttpResponse, HttpServer, Responder
};
    
use search::Search;
use std::error::Error;
use std::sync::{Arc, Mutex};

/**
 * Server routes and API
 */

#[get("/api/status")]
async fn status(data: web::Data<Arc<Mutex<Search>>>) -> impl Responder {
    web::Json(data.lock().unwrap().status())
}

/**
 * Server entry point
 */

#[actix_web::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Parse arguments
    let matches = App::new("kami")
        .version(env!("CARGO_PKG_VERSION"))
        .author("Justin Stanley <jtst@iastate.edu>")
        .about("A portable chess engine powered by reinforcement learning.")
        .arg(Arg::with_name("verbose")
            .short("v")
            .long("verbose")
            .help("Enables verbose logging"))
        .get_matches();

    // Initialize config
    let mut config = Config::new();

    config.set_default("verbose", false)?;
    config.set_default("data_dir", dirs::data_dir().unwrap().join("kami").to_str().unwrap())?;

    // Initialize logging
    if matches.is_present("verbose") {
        config.set("verbose", true)?;
    }

    std::env::set_var("RUST_LOG", "warn");

    if config.get_bool("verbose")? {
        std::env::set_var("RUST_LOG", std::env::var("RUST_LOG").unwrap() + ",debug");
    }

    pretty_env_logger::init();

    let searcher = Arc::new(Mutex::new(Search::new(&config)?));

    info!("Starting server on {}", BIND);

    // Start web service
    let srv = HttpServer::new(move ||
        actix_web::App::new()
            .data(searcher.clone())
            .service(status)
    )
    .bind(BIND)?
    .run();

    println!("Listening on {}", BIND);

    srv.await?;

    Ok(())
}

extern crate dirs;

use std::fs;

fn main() {
    let data_dir = dirs::data_dir().unwrap().join("kami");

    // Setup data directories
    fs::create_dir_all(data_dir).expect("Failed to initialize data directory");
}

/// Methods for manipulating kami paths and directories.
use crate::constants;
use crate::game::Game;

use dirs;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

/// Returns the path to the root data directory.
pub fn data_dir() -> Result<PathBuf, Box<dyn Error>> {
    let out = match dirs::data_dir() {
        Some(dd) => dd.join("kami"),
        None => return Err("No data directory available!".into()),
    };

    fs::create_dir_all(&out)?;
    Ok(out)
}

/// Returns the path to the archive directory.
pub fn archive_dir() -> Result<PathBuf, Box<dyn Error>> {
    let out = data_dir()?.join("archive");
    fs::create_dir_all(&out)?;

    Ok(out)
}

/// Returns the path to the current model.
pub fn model_dir() -> Result<PathBuf, Box<dyn Error>> {
    let out = data_dir()?.join("model");

    if out.exists() {
        if !out.is_dir() {
            return Err(format!(
                "Model path {} exists but is not a directory!",
                out.display()
            )
            .into());
        }
    }

    return Ok(out);
}

/// Returns the path to the current games directory.
pub fn games_dir() -> Result<PathBuf, Box<dyn Error>> {
    let out = get_generation_archive()?.join("games");

    fs::create_dir_all(&out)?;
    Ok(out)
}

/// Returns the current generation (regardless of completeness).
pub fn get_generation() -> Result<usize, Box<dyn Error>> {
    let p = data_dir()?.join("generation");

    if p.exists() {
        Ok(fs::read_to_string(&p)?.parse::<usize>()?)
    } else {
        fs::write(&p, b"0")?;
        Ok(0)
    }
}

/// Increments the generation number.
pub fn new_generation() -> Result<(), Box<dyn Error>> {
    let p = data_dir()?.join("generation");

    assert!(p.exists(), "No previous generation");

    fs::write(&p, get_generation()?.to_string())?;

    Ok(())
}

/// Gets the path to the current generation archive dir.
pub fn get_generation_archive() -> Result<PathBuf, Box<dyn Error>> {
    let p = archive_dir()?.join(format!("generation_{}", get_generation()?));

    fs::create_dir_all(&p)?;
    Ok(p)
}

/// Gets the path for the next game to be generated (or resumed), otherwise returns None
pub fn next_game() -> Result<Option<PathBuf>, Box<dyn Error>> {
    let gdir = games_dir()?;

    for i in 0..constants::TRAINING_SET_SIZE {
        let gpath = gdir.join(format!("{}.game", i));

        if gpath.exists() {
            if !gpath.is_file() {
                return Err(format!("Game {} exists but is not a file!", gpath.display()).into());
            }

            if !Game::load(&gpath)?.is_complete() {
                return Ok(Some(gpath));
            }

            continue;
        }

        return Ok(Some(gpath));
    }

    return Ok(None);
}

/// Gets the path for the next ELO eval game to be generated, otherwise returns None
pub fn next_elo_game() -> Result<Option<(PathBuf, usize)>, Box<dyn Error>> {
    let gdir = games_dir()?;

    for i in 0..constants::ELO_EVALUATION_NUM_GAMES {
        let gpath = gdir.join(format!("{}.elo_game", i));

        if gpath.exists() {
            if !gpath.is_file() {
                return Err(format!("Game {} exists but is not a file!", gpath.display()).into());
            }

            if !Game::load(&gpath)?.is_complete() {
                return Ok(Some((gpath, i)));
            }

            continue;
        }

        return Ok(Some((gpath, i)));
    }

    return Ok(None);
}

/// Returns all generated ELO games.
pub fn elo_game_set() -> Result<Vec<Game>, Box<dyn Error>> {
    let gdir = games_dir()?;
    let mut out = Vec::new();

    for i in 0..constants::ELO_EVALUATION_NUM_GAMES {
        let gpath = gdir.join(format!("{}.elo_game", i));

        out.push(Game::load(&gpath)?);
    }

    return Ok(out);
}

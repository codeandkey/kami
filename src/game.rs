use crate::input::trainbatch::TrainBatch;
use crate::position::Position;

use chess::ChessMove;
use rand::prelude::*;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::str::FromStr;

/// Holds data relevant to a single played game.
#[derive(Serialize, Deserialize)]
pub struct Game {
    mcts: Vec<Vec<(String, f32)>>,
    actions: Vec<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<f32>,
}

impl Game {
    /// Creates a new game at the starting position.
    pub fn new() -> Self {
        Game {
            actions: Vec::new(),
            mcts: Vec::new(),
            result: None,
        }
    }

    /// Loads a game from a path.
    pub fn load(p: &Path) -> Result<Game, Box<dyn Error>> {
        let g = serde_json::from_reader(BufReader::new(File::open(p)?))?;

        Ok(g)
    }

    /// Writes a game to a path.
    pub fn save(&self, p: &Path) -> Result<(), Box<dyn Error>> {
        serde_json::to_writer(BufWriter::new(File::create(p)?), self)?;
        Ok(())
    }

    /// Makes a move.
    pub fn make_move(&mut self, action: ChessMove, mcts: Vec<(ChessMove, f32)>) {
        self.actions.push(action.to_string());
        self.mcts
            .push(mcts.iter().map(|x| (x.0.to_string(), x.1)).collect());
    }

    /// Assigns the result for this game and allows it to be saved.
    pub fn finalize(&mut self, result: f32) {
        assert!(self.result.is_none());

        self.result = Some(result);
    }

    /// Tests if the game is completed.
    pub fn is_complete(&self) -> bool {
        self.result.is_some()
    }

    /// Gets the game moves in string format.
    pub fn to_string(&self) -> String {
        self.actions.join(" ")
    }

    /// Gets the game moves in vector format.
    pub fn get_actions(&self) -> Vec<ChessMove> {
        self.actions
            .iter()
            .map(|x| ChessMove::from_str(x).expect("bad move"))
            .collect()
    }

    /// Extends a TrainBatch from a random position in this game.
    pub fn add_to_batch(&self, tb: &mut TrainBatch) {
        let mut rng = thread_rng();
        let idx = rng.next_u32() as usize % self.actions.len();
        let mut pos = Position::new();

        // Make moves up to this point.
        for i in 0..idx {
            assert!(pos.make_move(ChessMove::from_str(&self.actions[i]).expect("bad move")));
        }

        // Build MCTS frame.
        let mut mcts = [0.0; 4096];

        for (mv_str, val) in &self.mcts[idx] {
            let mv = ChessMove::from_str(mv_str).expect("bad move");
            mcts[mv.get_source().to_index() * 64 + mv.get_dest().to_index()] = *val;
        }

        tb.add(&pos, &mcts, self.result.unwrap());
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::io::{Read, Write};

    /// Tests a game can be initialized.
    #[test]
    fn game_can_initialize() {
        Game::new();
    }

    /// Tests a game can be loaded from a file.
    #[test]
    fn game_can_load() {
        let gdir = tempfile::tempdir()
            .expect("failed creating data dir")
            .into_path();
        let gpath = gdir.join("test.json");

        File::create(&gpath).expect("failed creating game file").write(r#"{"mcts":[[["a2a3",0.055456806],["a2a4",0.031994313],["b2b3",0.04372556],["b2b4",0.036437966],["c2c3",0.033771776],["c2c4",0.062211163],["d2d3",0.05083541],["d2d4",0.03768219],["e2e3",0.030750088],["e2e4",0.05954497],["f2f3",0.049591184],["f2f4",0.0627444],["g2g3",0.042125843],["g2g4",0.057767507],["h2h3",0.060611445],["h2h4",0.05954497],["b1a3",0.062388908],["b1c3",0.0419481],["g1f3",0.05901173],["g1h3",0.06185567]]],"actions":["f2f4"],"result":null}"#.as_bytes()
        ).expect("write failed");

        let g = Game::load(&gpath).expect("failed loading game");

        assert_eq!(g.is_complete(), false);
        assert_eq!(g.get_actions().len(), 1);
        assert_eq!(g.get_actions()[0].to_string(), "f2f4".to_string());
    }

    /// Tests a game can be written to a file.
    #[test]
    fn game_can_write() {
        let gdir = tempfile::tempdir()
            .expect("failed creating data dir")
            .into_path();
        let gpath = gdir.join("test.json");

        let mut g = Game::new();

        let mcts_data = vec![
            ("a2a3",0.055456806),
            ("a2a4",0.031994313),
            ("b2b3",0.04372556),
            ("b2b4",0.036437966),
            ("c2c3",0.033771776),
            ("c2c4",0.062211163),
            ("d2d3",0.05083541),
            ("d2d4",0.03768219),
            ("e2e3",0.030750088),
            ("e2e4",0.05954497),
            ("f2f3",0.049591184),
            ("f2f4",0.0627444),
            ("g2g3",0.042125843),
            ("g2g4",0.057767507),
            ("h2h3",0.060611445),
            ("h2h4",0.05954497),
            ("b1a3",0.062388908),
            ("b1c3",0.0419481),
            ("g1f3",0.05901173),
            ("g1h3",0.06185567)
        ];

        let mcts_data = mcts_data.into_iter().map(|(a, b)| (ChessMove::from_str(a).expect("bad move"), b)).collect();

        g.make_move(
            ChessMove::from_str("f2f4").expect("bad move"),
            mcts_data,
        );

        g.save(&gpath).expect("write failed");

        let mut contents = Vec::new();
        File::open(&gpath).expect("read failed").read_to_end(&mut contents).expect("read failed");

        let contents = String::from_utf8(contents).unwrap();

        assert_eq!(
            contents,
            r#"{"mcts":[[["a2a3",0.055456806],["a2a4",0.031994313],["b2b3",0.04372556],["b2b4",0.036437966],["c2c3",0.033771776],["c2c4",0.062211163],["d2d3",0.05083541],["d2d4",0.03768219],["e2e3",0.030750088],["e2e4",0.05954497],["f2f3",0.049591184],["f2f4",0.0627444],["g2g3",0.042125843],["g2g4",0.057767507],["h2h3",0.060611445],["h2h4",0.05954497],["b1a3",0.062388908],["b1c3",0.0419481],["g1f3",0.05901173],["g1h3",0.06185567]]],"actions":["f2f4"]}"#.to_string()
        );
    }

    /// Tests a game can finalize.
    #[test]
    fn game_can_finalize() {
        let mut g = Game::new();

        assert!(!g.is_complete());
        g.finalize(0.0);
        assert!(g.is_complete());
    }

    /// Tests a game can be converted to string.
    #[test]
    fn game_can_get_string() {
        let mut g = Game::new();
        
        assert_eq!(g.to_string(), "".to_string());

        g.make_move(
            ChessMove::from_str("f2f4").expect("bad move"),
            Vec::new(),
        );

        assert_eq!(g.to_string(), "f2f4".to_string());
    }

    /// Tests a game can be added to a training batch.
    #[test]
    fn game_can_add_to_trainbatch() {
        let mut g = Game::new();

        let mcts_data = vec![
            ("a2a3",0.055456806),
            ("a2a4",0.031994313),
            ("b2b3",0.04372556),
            ("b2b4",0.036437966),
            ("c2c3",0.033771776),
            ("c2c4",0.062211163),
            ("d2d3",0.05083541),
            ("d2d4",0.03768219),
            ("e2e3",0.030750088),
            ("e2e4",0.05954497),
            ("f2f3",0.049591184),
            ("f2f4",0.0627444),
            ("g2g3",0.042125843),
            ("g2g4",0.057767507),
            ("h2h3",0.060611445),
            ("h2h4",0.05954497),
            ("b1a3",0.062388908),
            ("b1c3",0.0419481),
            ("g1f3",0.05901173),
            ("g1h3",0.06185567)
        ];

        let mcts_data = mcts_data.into_iter().map(|(a, b)| (ChessMove::from_str(a).expect("bad move"), b)).collect();

        g.make_move(
            ChessMove::from_str("f2f4").expect("bad move"),
            mcts_data,
        );

        g.finalize(1.0);

        let mut tb = TrainBatch::new(1);

        g.add_to_batch(&mut tb);

        assert_eq!(tb.get_inner().get_size(), 1);
        assert_eq!(tb.get_results(), &[1.0]);
    }
}

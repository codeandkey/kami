/**
 * Input layer type.
 */
 
use chess::{Board, Color};
use crate::inputframe::InputFrame;

pub const PLY_FRAME_SIZE: usize = 14;
pub const PLY_FRAME_COUNT: usize = 6;
pub const SQUARE_HEADER_SIZE: usize = 24;
pub const SQUARE_BITS: usize = SQUARE_HEADER_SIZE + PLY_FRAME_SIZE * PLY_FRAME_COUNT;
pub const COUNTER_BITS: usize = 20;

pub struct Input {
    counters: [f32; COUNTER_BITS],
    castle_rights: [f32; 4],
    frames: Vec<InputFrame>,
}

impl Input {
    pub fn new(b: &Board, move_number: usize, halfmove_clock: usize) -> Self {
        let mut inp = Input {
            counters: [0.0; COUNTER_BITS],
            castle_rights: [1.0; 4],
            frames: vec![InputFrame::new(b, 0)],
        };

        inp.write_headers(b, move_number, halfmove_clock);
        inp
    }

    pub fn write_headers(&mut self, b: &Board, move_num: usize, hmc: usize) {
        // Write full move number
        for i in 0..14 {
            self.counters[i] = ((move_num >> i) & 0x1) as f32;
        }

        // Write halfmove clock
        for i in 0..6 {
            self.counters[i + 14] = ((hmc >> i) & 0x1) as f32;
        }

        // Write castling rights
        let wrights = b.castle_rights(Color::White);
        let brights = b.castle_rights(Color::Black);

        self.castle_rights[0] = wrights.has_kingside() as i32 as f32;
        self.castle_rights[1] = wrights.has_queenside() as i32 as f32;
        self.castle_rights[2] = brights.has_kingside() as i32 as f32;
        self.castle_rights[3] = brights.has_queenside() as i32 as f32;
    }

    pub fn push_frame(&mut self, frame: InputFrame) {
        self.frames.push(frame);
    }

    pub fn pop_frame(&mut self) {
        self.frames.pop().expect("pop frame failed");
    }
}
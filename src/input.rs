/**
 * Input layer type.
 */
 
use crate::inputframe::InputFrame;
use crate::net;

use chess::{Board, Color};

#[derive(Clone)]
pub struct Input {
    headers: [f32; 24],
    frames: Vec<InputFrame>,
}

impl Input {
    pub fn new(b: &Board, move_number: u8, halfmove_clock: u8) -> Self {
        let mut inp = Input {
            headers: [0.0; 24],
            frames: vec![InputFrame::new(b, 0)],
        };

        inp.write_headers(b, move_number, halfmove_clock);
        inp
    }

    pub fn write_headers(&mut self, b: &Board, move_num: u8, hmc: u8) {
        // Write full move number
        for i in 0..8 {
            self.headers[i] = ((move_num >> i) & 0x1) as f32;
        }

        // Write halfmove clock
        for i in 0..6 {
            self.headers[i + 14] = ((hmc >> i) & 0x1) as f32;
        }

        // Write castling rights
        let wrights = b.castle_rights(Color::White);
        let brights = b.castle_rights(Color::Black);

        self.headers[20] = wrights.has_kingside() as i32 as f32;
        self.headers[21] = wrights.has_queenside() as i32 as f32;
        self.headers[22] = brights.has_kingside() as i32 as f32;
        self.headers[23] = brights.has_queenside() as i32 as f32;
    }

    pub fn push_frame(&mut self, frame: InputFrame) {
        self.frames.push(frame);
    }

    pub fn pop_frame(&mut self) {
        self.frames.pop().expect("pop frame failed");
    }

    pub fn get_headers(&self) -> &[f32; 24] {
        &self.headers
    }

    pub fn get_frames(&self) -> impl Iterator<Item = &InputFrame> {
        self.frames.iter().rev().take(net::PLY_FRAME_COUNT).rev()
    }
}
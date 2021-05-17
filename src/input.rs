/**
 * Input layer type.
 */

use chess::Board;

pub const PLY_FRAME_SIZE: usize = 14;
pub const PLY_FRAME_COUNT: usize = 6;
pub const SQUARE_BITS: usize = 108;
pub const SQUARE_HEADER_SIZE: usize = 24;

pub struct Input {
    layer: Vec<f32>,
    hist_frames: Vec<Vec<Vec<Vec<f32>>>>,
}

impl Input {
    pub fn new() -> Self {
        let mut layer = Vec::new();
        layer.resize(SQUARE_BITS * 64, 0.0);

        let mut hist_frames: Vec<Vec<Vec<Vec<f32>>>> = Vec::new();

        hist_frames.resize_with(8, Vec::new);

        for i in 0..8 {
            hist_frames[i].resize_with(8, Vec::new);
        }

        Input {
            layer: layer,
            hist_frames: hist_frames,
        }
    }

    pub fn write_header(&mut self, rank: usize, file: usize, move_num: usize, hmc: usize, our_ks: bool, our_qs: bool, their_ks: bool, their_qs: bool) {
        let offset = rank * (8 * SQUARE_BITS) + file * SQUARE_BITS;

        // Write full move number
        for i in 0..14 {
            self.layer[offset + i] = ((move_num >> i) & 0x1) as f32;
        }

        // Write halfmove clock
        for i in 0..6 {
            self.layer[offset + 14 + i] = ((hmc >> i) & 0x1) as f32;
        }

        // Write castling rights
        self.layer[offset + 20] = if our_ks { 1.0 } else { 0.0 };
        self.layer[offset + 21] = if our_qs { 1.0 } else { 0.0 };
        self.layer[offset + 22] = if their_ks { 1.0 } else { 0.0 };
        self.layer[offset + 23] = if their_qs { 1.0 } else { 0.0 };
    }

    pub fn write_frame(&mut self, r: usize, f: usize, pbit: usize) {
        self.layer[r * (8 * SQUARE_BITS) + f * SQUARE_BITS + 24 + pbit] = 1.0;
    }

    pub fn clear_frame(&mut self, r: usize, f: usize) {
        let offset = r * (8 * SQUARE_BITS) + f * SQUARE_BITS + 24;
        self.layer[offset .. offset + 12].fill(0.0);
    }

    pub fn push_frames(&mut self) {
        for r in 0..8 {
            for f in 0..8 {
                let start = r * (8 * SQUARE_BITS) + f * SQUARE_BITS + 24;
                let end = start + PLY_FRAME_SIZE * PLY_FRAME_COUNT;

                // Save frame before rotating
                self.hist_frames[r][f].push(self.layer[end - PLY_FRAME_SIZE .. end].to_vec());

                self.layer[start .. end].rotate_right(PLY_FRAME_SIZE);
            }
        }
    }

    pub fn pop_frames(&mut self) {
        for r in 0..8 {
            for f in 0..8 {
                let start = r * (8 * SQUARE_BITS) + f * SQUARE_BITS + 24;
                let end = start + PLY_FRAME_SIZE * PLY_FRAME_COUNT;

                self.layer[start .. end].rotate_left(PLY_FRAME_SIZE);

                // Restore frame if there is one, otherwise fill with 0
                if self.hist_frames.len() > 0 {
                    self.layer[end - PLY_FRAME_SIZE .. end].copy_from_slice(&self.hist_frames[r][f].pop().unwrap());
                } else {
                    self.layer[end - PLY_FRAME_SIZE .. end].fill(0.0);
                }
            }
        }
    }
}
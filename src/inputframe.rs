/**
 * Input board frame
 * 
 * Includes piece and repetition bits.
 */

use crate::net;
use chess::{Board, Square, Rank, File};

#[derive(Clone)]
pub struct InputFrame {
    frame: [f32; net::PLY_FRAME_SIZE * 64], // might need to flatten
}

impl InputFrame {
    pub fn new(b: &Board, reps: u8) -> Self {
        let mut new_iframe = InputFrame {
            frame: [0.0; net::PLY_FRAME_SIZE * 64],
        };

        let rbitlow = (reps & 1) as f32;
        let rbithigh = ((reps >> 1) & 1) as f32;

        // Should be the _only_ time we need to iterate over squares, ideally
        for r in 0..8 {
            for f in 0..8 {
                let sq = Square::make_square(Rank::from_index(r), File::from_index(f));
                let p = b.piece_on(sq);

                new_iframe.write(r, f, 12, rbitlow);
                new_iframe.write(r, f, 13, rbithigh);

                if let Some(pc) = p {
                    let mut pbit = pc.to_index();
                    pbit += b.color_on(sq).unwrap().to_index() * 6;

                    new_iframe.write(r, f, pbit, 1.0);
                }
            }
        }

        new_iframe
    }

    pub fn get_data(&self) -> &[f32; net::PLY_FRAME_SIZE * 64] {
        &self.frame
    }

    pub fn get_square(&self, rank: usize, file: usize) -> &[f32] {
        &self.frame[rank * (14 * 8) + file * 14 .. rank * (14 * 8) + (file + 1) * 14]
    }

    fn write(&mut self, rank: usize, file: usize, bit: usize, value: f32) {
        self.frame[rank * (14 * 8) + file * 14 + bit] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_input_frame() {
        let b = chess::Board::default();
        let iframe = InputFrame::new(&b, 0);

        let data = iframe.get_data();

        // check inner bits of board are all 0
        for r in 2..6 {
            for f in 0..8 {
                assert_eq!(iframe.get_square(r, f), [0.0; 14]);
            }
        }

        // check white pawns
        for f in 0..8 {
            assert_eq!(iframe.get_square(1, f), [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        }

        // check white knights
        assert_eq!(iframe.get_square(0, 1), [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(iframe.get_square(0, 6), [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // check white bishops
        assert_eq!(iframe.get_square(0, 2), [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(iframe.get_square(0, 5), [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // check white rooks
        assert_eq!(iframe.get_square(0, 0), [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(iframe.get_square(0, 7), [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // check white queen
        assert_eq!(iframe.get_square(0, 3), [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // check white king
        assert_eq!(iframe.get_square(0, 4), [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // check black pawns
        for f in 0..8 {
            assert_eq!(iframe.get_square(6, f), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        }

        // check black knights
        assert_eq!(iframe.get_square(7, 1), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(iframe.get_square(7, 6), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // check black bishops
        assert_eq!(iframe.get_square(7, 2), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(iframe.get_square(7, 5), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // check black rooks
        assert_eq!(iframe.get_square(7, 0), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(iframe.get_square(7, 7), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

        // check black queen
        assert_eq!(iframe.get_square(7, 3), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);

        // check black king
        assert_eq!(iframe.get_square(7, 4), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    
}
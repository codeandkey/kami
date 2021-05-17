/**
 * Input board frame
 * 
 * Includes piece and repetition bits.
 */

use chess::{BitBoard, Board};

pub struct InputFrame {
    frame: [[[f32; 14]; 8]; 8], // might need to flatten
}

impl InputFrame {
    pub fn new(b: &Board, reps: usize) -> Self {
        let mut frame = [[[0.0; 14]; 8]; 8];
        let rbitlow = (reps & 1) as f32;
        let rbithigh = ((reps >> 1) & 1) as f32;

        // Should be the _only_ time we need to iterate over squares, ideally
        for sq in BitBoard::new(u64::MAX) {
            let (r, f) = (sq.get_rank().to_index(), sq.get_file().to_index());
            let p = b.piece_on(sq);

            frame[r][f][12] = rbitlow;
            frame[r][f][13] = rbithigh;

            if let Some(pc) = p {
                let mut pbit = pc.to_index();
                pbit += b.color_on(sq).unwrap().to_index() * 6;

                frame[r][f][pbit] = 1.0;
            }
        }

        InputFrame {
            frame: frame,
        }
    }

    pub fn get_data(&self) -> &[[[f32; 14]; 8]; 8] {
        &self.frame
    }
}
use crate::input::Input;
use chess::{Board, ChessMove, Color, Square, Piece, Rank, File, MoveGen};

pub struct Position {
    hist_boards: Vec<Board>,
    hmc_bits: Vec<usize>,
    white_input: Input,
    black_input: Input,
}

impl Position {
    pub fn new() -> Self {
        let mut p = Position {
            hist_boards: vec![Board::default()],
            white_input: Input::new(),
            black_input: Input::new(),
            hmc_bits: vec![0],
        };

        p.write_frame();
        p
    }

    pub fn generate_moves(&self) -> Vec<chess::ChessMove> {
        let gen = MoveGen::new_legal(self.hist_boards.last().unwrap());
        let mut output = Vec::new();

        for current in gen {
            output.push(current);
        }

        output
    }

    pub fn make_move(&mut self, mv: ChessMove) -> Result<(), chess::Error> {
        // Set new hmc
        let src_piece = self.hist_boards.last().unwrap().piece_on(mv.get_source());
        let dst_piece = self.hist_boards.last().unwrap().piece_on(mv.get_dest());

        if src_piece.unwrap() == Piece::Pawn || dst_piece.is_some() {
            self.hmc_bits.push(0);
        } else {
            let last_hmc = *self.hmc_bits.last().unwrap();
            self.hmc_bits.push(last_hmc + 1);
        }

        // Make move on board
        self.hist_boards.push(self.hist_boards.last().unwrap().make_move_new(mv));

        // Push input frames
        self.white_input.push_frames();
        self.black_input.push_frames();

        // Write current frame
        self.write_frame();

        Ok(())
    }

    pub fn unmake_move(&mut self) {
        // Pop board state
        self.hist_boards.pop();

        // Pop history frames
        self.white_input.pop_frames();
        self.black_input.pop_frames();

        // Write current frame
        self.write_frame();
    }

    fn write_frame(&mut self) {
        let move_number = (self.hist_boards.len() - 1) / 2;
        let halfmove_clock = *self.hmc_bits.last().unwrap();

        let board = self.hist_boards.last().unwrap();
        let wrights = board.castle_rights(Color::White);
        let brights = board.castle_rights(Color::Black);

        let wks = wrights.has_kingside();
        let wqs = wrights.has_queenside();
        let bks = brights.has_kingside();
        let bqs = brights.has_queenside();

        // Iterate board squares once
        for r in 0..8 {
            for f in 0..8 {
                // Get piece type, color if there is one
                let sq = Square::make_square(Rank::from_index(r), File::from_index(f));
                let pc = board.piece_on(sq);
                let col = board.color_on(sq);

                // Write white header at destination (r, f)
                self.white_input.write_header(r, f, move_number, halfmove_clock, wks, wqs, bks, bqs);

                // Write black header at destination (7 - r, 7 - f) to mirror
                self.black_input.write_header(7 - r, 7 - f, move_number, halfmove_clock, wks, wqs, bks, bqs);

                // Write piece bits if there is one
                if let Some(p) = pc {
                    if col.unwrap() == Color::White {
                        self.white_input.write_frame(r, f, p.to_index());
                        self.black_input.write_frame(7 - r, 7 - f, 6 + p.to_index());
                    } else {
                        self.white_input.write_frame(r, f, 6 + p.to_index());
                        self.black_input.write_frame(7 - r, 7 - f, p.to_index());
                    }
                } else {
                    // No piece, just clear the frame
                    self.white_input.clear_frame(r, f);
                    self.black_input.clear_frame(7 - r, 7 - f);
                }
            }
        }
    }

    pub fn get_input(&self) -> &Input {
        match self.hist_boards.last().unwrap().side_to_move() {
            Color::White => &self.white_input,
            Color::Black => &self.black_input,
        }
    }

    pub fn get_fen(&self) -> String {
        self.hist_boards.last().unwrap().to_string()
    }
}
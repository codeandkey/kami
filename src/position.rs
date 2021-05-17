use crate::input::Input;
use crate::inputframe::InputFrame;
use chess::{Board, ChessMove, Color, Square, Piece, Rank, File, MoveGen};

struct State {
    b: Board,
    halfmove_clock: usize,
    move_number: usize,
    repetitions: usize,
}

pub struct Position {
    states: Vec<State>,
    input: Input,
}

impl Position {
    pub fn new() -> Self {
        let b = Board::default();

        Position {
            states: vec![State {
                b: b,
                halfmove_clock: 0,
                move_number: 1,
                repetitions: 0,
            }],
            input: Input::new(&b, 1, 0),
        }
    }

    fn top(&self) -> &State {
        &self.states.last().unwrap()
    }

    pub fn iterate_moves(&self) -> impl Iterator<Item = chess::ChessMove> {
        MoveGen::new_legal(&self.top().b)
    }

    pub fn make_move(&mut self, mv: ChessMove) -> Result<(), chess::Error> {
        // Find next halfmove clock
        let src_piece = self.top().b.piece_on(mv.get_source());
        let dst_piece = self.top().b.piece_on(mv.get_dest());
        let mut next_halfmove_clock = self.top().halfmove_clock + 1;

        if src_piece.unwrap() == Piece::Pawn || dst_piece.is_some() {
            next_halfmove_clock = 0;
        }

        // Find next move number
        let mut next_move_number = self.top().move_number;

        if self.top().b.side_to_move() == Color::Black {
            next_move_number += 1;
        }

        // Make move, build next board
        let next_board = self.top().b.make_move_new(mv);

        // Count repetitions
        let current_hash = next_board.get_hash();
        let mut num_repetitions = 0;

        for s in &self.states {
            if s.b.get_hash() == current_hash {
                num_repetitions += 1;
            }
        }

        // Build next state
        let next_state = State {
            b: next_board,
            halfmove_clock: next_halfmove_clock,
            move_number: next_move_number,
            repetitions: num_repetitions,
        };

        // Update input header
        self.input.write_headers(&next_board, next_state.move_number, next_state.halfmove_clock);

        // Push new frame
        self.input.push_frame(InputFrame::new(&next_board, next_state.repetitions));

        // Push state to history
        self.states.push(next_state);

        Ok(())
    }

    pub fn unmake_move(&mut self) {
        // Pop board state
        self.states.pop();

        let top = self.states.last().unwrap();

        // Update input header
        self.input.write_headers(&top.b, top.move_number, top.halfmove_clock);

        // Pop input frame
        self.input.pop_frame();
    }

    pub fn get_input(&self) -> &Input {
        &self.input
    }

    pub fn get_fen(&self) -> String {
        self.top().b.to_string()
    }
}
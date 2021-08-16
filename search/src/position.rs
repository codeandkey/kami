use crate::consts;
use chess::{Board, ChessMove, Color, File, MoveGen, Piece, Rank, Square};

#[derive(Clone)]
struct State {
    pub b: Board,
    pub halfmove_clock: u8,
    pub move_number: u8,
    pub hash: u64,
    header: [f32; consts::HEADER_SIZE],
}

impl State {
    /// Returns an instance of State representing the starting position.
    pub fn initial() -> Self {
        let b = Board::default();

        State {
            halfmove_clock: 0,
            move_number: 1,
            header: State::gen_headers(0, 1, &b),
            hash: b.get_hash(),
            b: b,
        }
    }

    /// Returns a Some(State) if an action can be performed, or None otherwise.
    pub fn next(prev: &Self, mv: ChessMove) -> Option<Self> {
        if !prev.b.legal(mv) {
            return None;
        }

        // Find next halfmove clock
        let src_piece = prev.b.piece_on(mv.get_source());
        let dst_piece = prev.b.piece_on(mv.get_dest());
        let mut next_halfmove_clock = prev.halfmove_clock + 1;

        if src_piece.unwrap() == Piece::Pawn || dst_piece.is_some() {
            next_halfmove_clock = 0;
        }

        // Find next move number
        let mut next_move_number = prev.move_number;

        if prev.b.side_to_move() == Color::Black {
            next_move_number += 1;
        }

        // Make move, build next board
        let next_board = prev.b.make_move_new(mv);

        // Build next state
        Some(State {
            header: State::gen_headers(next_halfmove_clock, next_move_number, &next_board),
            hash: next_board.get_hash(),
            b: next_board,
            halfmove_clock: next_halfmove_clock,
            move_number: next_move_number,
        })
    }

    pub fn get_header(&self) -> &[f32] {
        &self.header
    }

    /// Generates an input header given a move number, halfmove clock and board (castle rights).
    fn gen_headers(
        halfmove_clock: u8,
        move_number: u8,
        b: &Board,
    ) -> [f32; consts::HEADER_SIZE] {
        let mut hdr = [0.0; consts::HEADER_SIZE];

        // Write full move number
        for i in 0..8 {
            hdr[i] = ((move_number >> i) & 0x1) as f32;
        }

        // Write halfmove clock
        for i in 0..6 {
            hdr[i + 8] = ((halfmove_clock >> i) & 0x1) as f32;
        }

        // Write castling rights
        let mrights = b.my_castle_rights();
        let trights = b.their_castle_rights();

        hdr[14] = mrights.has_kingside() as i32 as f32;
        hdr[15] = mrights.has_queenside() as i32 as f32;
        hdr[16] = trights.has_kingside() as i32 as f32;
        hdr[17] = trights.has_queenside() as i32 as f32;

        hdr
    }
}

#[derive(Clone)]
pub struct Position {
    states: Vec<State>,
    reps: Vec<u8>,
    wframes: Vec<f32>,
    bframes: Vec<f32>,
}

impl Position {
    /// Creates a new Position instance loaded with the starting position.
    pub fn new() -> Self {
        // Fill initial frames with 0s to ensure the get_frames() slice is always the same
        let mut initial_frames = Vec::new();
        initial_frames.resize(consts::FRAME_COUNT * 64 * consts::FRAME_SIZE, 0.0);

        let mut p = Position {
            states: vec![State::initial()],
            wframes: initial_frames.clone(),
            bframes: initial_frames,
            reps: vec![0],
        };

        p.push(0);
        p
    }

    /// Gets the most recent game state.
    fn top(&self) -> &State {
        &self.states.last().unwrap()
    }

    /// Gets the current side to move.
    pub fn side_to_move(&self) -> Color {
        self.top().b.side_to_move()
    }

    /// Returns an iterator over the legal moves in this Position.
    pub fn iterate_moves(&self) -> impl Iterator<Item = ChessMove> {
        MoveGen::new_legal(&self.top().b)
    }

    /// Returns a vector containing all legal moves for this position.
    pub fn generate_moves(&self) -> Vec<ChessMove> {
        self.iterate_moves().collect()
    }

    /// Makes a move on the board.
    ///
    /// Returns true if the move was sucessfully performed.
    /// Returns false if the move is illegal and no move was made.
    pub fn make_move(&mut self, mv: ChessMove) -> bool {
        let current_hash: u64;

        if let Some(next) = State::next(self.top(), mv) {
            current_hash = next.hash;
            self.states.push(next);
        } else {
            return false;
        }

        // Count repetitions
        let mut reps = 0;

        for s in &self.states {
            if s.hash == current_hash {
                reps += 1;
            }
        }

        // Iterate over squares and build next input frame.
        self.push(reps);

        return true;
    }

    /// Returns Some(result) if the game is over.
    /// Result is 1 if white wins, -1 if black wins, and 0 if the game is a draw.
    /// Returns None if the game is still ongoing.
    pub fn is_game_over(&self) -> Option<f64> {
        let b = self.top().b;

        let status = match b.status() {
            chess::BoardStatus::Checkmate => match b.side_to_move() {
                chess::Color::White => Some(-1.0),
                chess::Color::Black => Some(1.0),
            },
            chess::BoardStatus::Stalemate => Some(0.0),
            chess::BoardStatus::Ongoing => None,
        };

        if status.is_some() {
            return status;
        }

        // Check for insufficient material
        let all_pieces = b.color_combined(Color::White) | b.color_combined(Color::Black);

        // K(N)(B)vK(N)(B)
        if b.pieces(Piece::King) | b.pieces(Piece::Knight) | b.pieces(Piece::Bishop) == all_pieces && (b.pieces(Piece::Knight) | b.pieces(Piece::Bishop)).popcnt() < 3 {
            return Some(0.0);
        }

        // Check for 50-move rule
        if self.top().halfmove_clock >= 50 {
            return Some(0.0);
        }

        // Test for threefold repetition
        let current_hash = self.top().hash;
        let mut reps = 0;

        for s in &self.states {
            if s.hash == current_hash {
                reps += 1;
            }
        }

        if reps >= 3 {
            return Some(0.0);
        }

        return None;
    }

    /// Unmakes the last move made on this position.
    /// Panics if there is no move to unmake.
    pub fn unmake_move(&mut self) {
        // Pop board state
        self.states.pop();

        // Pop rep state
        self.reps.pop();

        // Shorten frame vector
        self.wframes
            .drain(self.wframes.len() - (consts::FRAME_SIZE * 64)..);
        self.bframes
            .drain(self.bframes.len() - (consts::FRAME_SIZE * 64)..);
    }

    /// Returns a reference to the current input frames.
    pub fn get_frames(&self) -> &[f32] {
        match self.top().b.side_to_move() {
            Color::White => &self.wframes[self.wframes.len() - consts::TOTAL_FRAMES_SIZE..],
            Color::Black => &self.bframes[self.bframes.len() - consts::TOTAL_FRAMES_SIZE..],
        }
    }

    /// Returns a reference to the current input headers.
    pub fn get_headers(&self) -> &[f32] {
        &self.top().get_header()
    }

    /// Returns a slice with the legal move mask for this position as well as a list of legal moves.
    pub fn get_lmm(&self) -> ([f32; 4096], Vec<ChessMove>) {
        let mut lmm = [0.0; 4096];
        let moves = self.generate_moves();

        match self.top().b.side_to_move() {
            Color::White => {
                for mv in &moves {
                    lmm[mv.get_source().to_index() * 64 + mv.get_dest().to_index()] = 1.0;
                }
            }
            Color::Black => {
                for mv in &moves {
                    lmm[(63 - mv.get_source().to_index()) * 64 + (63 - mv.get_dest().to_index())] =
                        1.0;
                }
            }
        }

        assert_ne!(moves.len(), 0, "in position {}", self.get_fen());

        (lmm, moves)
    }

    /// Returns the current position FEN.
    pub fn get_fen(&self) -> String {
        self.top().b.to_string()
    }

    /// Returns the current ply count.
    pub fn ply(&self) -> usize {
        self.states.len() - 1
    }

    /// Pushes the current board/rep state onto the frames layer.
    fn push(&mut self, reps: usize) {
        let b = self.top().b;
        let last_len = self.wframes.len();

        assert_eq!(self.wframes.len(), self.bframes.len());

        self.wframes
            .resize(last_len + consts::FRAME_SIZE * 64, 0.0);
        self.bframes
            .resize(last_len + consts::FRAME_SIZE * 64, 0.0);

        let wdst = &mut self.wframes[last_len..];
        let bdst = &mut self.bframes[last_len..];

        let rbitlow = (reps & 1) as f32;
        let rbithigh = ((reps >> 1) & 1) as f32;

        // Should be the _only_ time we need to iterate over squares, ideally
        for r in 0..8 {
            for f in 0..8 {
                let sq = Square::make_square(Rank::from_index(r), File::from_index(f));
                let p = b.piece_on(sq);

                // White input update
                let offset = r * (consts::FRAME_SIZE * 8) + f * consts::FRAME_SIZE;
                let dst_square_frame = &mut wdst[offset..offset + consts::FRAME_SIZE];

                dst_square_frame[12] = rbitlow;
                dst_square_frame[13] = rbithigh;

                if let Some(pc) = p {
                    let mut pbit = pc.to_index();
                    pbit += b.color_on(sq).unwrap().to_index() * 6;

                    dst_square_frame[pbit] = 1.0;
                }

                // Black input update
                let boffset =
                    (7 - r) * (consts::FRAME_SIZE * 8) + (7 - f) * consts::FRAME_SIZE;
                let bdst_square_frame = &mut bdst[boffset..boffset + consts::FRAME_SIZE];

                bdst_square_frame[12] = rbitlow;
                bdst_square_frame[13] = rbithigh;

                if let Some(pc) = p {
                    let mut pbit = pc.to_index() + 6;
                    pbit -= b.color_on(sq).unwrap().to_index() * 6;

                    bdst_square_frame[pbit] = 1.0;
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::str::FromStr;

    /// Tests a position can be initialized.
    #[test]
    fn position_can_initialize() {
        Position::new();
    }

    /// Tests a position can make legal moves.
    #[test]
    fn position_can_make_moves() {
        let mut p = Position::new();

        assert!(p.make_move(ChessMove::from_str("e2e4").expect("Failed to parse move.")));
        assert!(p.make_move(ChessMove::from_str("e7e5").expect("Failed to parse move.")));
        assert!(p.make_move(ChessMove::from_str("e1e2").expect("Failed to parse move.")));
        assert!(p.make_move(ChessMove::from_str("e8e7").expect("Failed to parse move.")));
    }

    /// Tests a position can unmake moves.
    #[test]
    fn position_can_unmake_moves() {
        let mut p = Position::new();

        assert!(p.make_move(ChessMove::from_str("e2e4").expect("Failed to parse move.")));
        assert!(p.make_move(ChessMove::from_str("e7e5").expect("Failed to parse move.")));
        assert!(p.make_move(ChessMove::from_str("e1e2").expect("Failed to parse move.")));
        assert!(p.make_move(ChessMove::from_str("e8e7").expect("Failed to parse move.")));

        for _ in 0..4 {
            p.unmake_move();
        }
    }

    /// Tests a position can return a FEN.
    #[test]
    fn position_can_get_fen() {
        let p = Position::new();

        assert_eq!(
            p.get_fen(),
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        );
    }

    /// Tests the position can iterate over legal moves.
    #[test]
    fn position_can_iterate_moves() {
        let p = Position::new();
        let moves = p.generate_moves();

        let expected_moves: Vec<ChessMove> = [
            "a2a4", "b2b4", "c2c4", "d2d4", "e2e4", "f2f4", "g2g4", "h2h4", "a2a3", "b2b3", "c2c3",
            "d2d3", "e2e3", "f2f3", "g2g3", "h2h3", "b1a3", "b1c3", "g1f3", "g1h3",
        ]
        .iter()
        .map(|m| ChessMove::from_str(m).expect("Failed parsing move"))
        .collect();

        for m in moves {
            assert!(expected_moves.contains(&m));
        }
    }

    /// Tests the initial position input layer.
    #[test]
    fn position_initial_input_layer_is_correct() {
        let p = Position::new();

        let frames = p.get_frames();
        let headers = p.get_headers();

        println!("frames: {:?}", frames);
        println!("headers: {:?}", headers);

        assert_eq!(
            headers,
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Move number = 1
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Halfmove clock = 0
                1.0, 1.0, 1.0, 1.0, // Castling = all
            ]
        );

        // First (n-1) frames should be 0.
        assert_eq!(
            frames[0..(consts::FRAME_SIZE * 64 * (consts::FRAME_COUNT - 1))],
            [0.0; (consts::FRAME_SIZE * 64 * (consts::FRAME_COUNT - 1))]
        );

        // Last frame should represent the starting board.
        assert_eq!(
            &frames[(consts::FRAME_SIZE * 64 * (consts::FRAME_COUNT - 1))..],
            [
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
            ]
        );
    }
}
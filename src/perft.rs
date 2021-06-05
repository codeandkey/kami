/**
 * Perft testing utility, for measuring performance of move-making and updating input layers
 */
use crate::position::Position;
use std::time::SystemTime;

pub fn perft(root: &mut Position, maxdepth: usize) {
    println!("Starting perft test from position {}", root.get_fen());

    for depth in 0..(maxdepth + 1) {
        println!("Starting perft with depth {}", depth);

        let starting_fen = root.get_fen();

        let time_before = SystemTime::now();
        let nodes = perft_sub(root, depth);
        let time_after = SystemTime::now();

        // Check the FEN hasn't changed
        let ending_fen = root.get_fen();
        assert_eq!(starting_fen, ending_fen);

        let total_ms = time_after.duration_since(time_before).unwrap().as_millis();

        println!(
            "Finished perft {}, {} nodes, {} ms, {} nps",
            depth,
            nodes,
            total_ms,
            (nodes as u128 * 1000) / (total_ms + 1)
        );
    }
}

fn perft_sub(root: &mut Position, depth: usize) -> usize {
    if depth <= 0 {
        return 1;
    }

    // Iterate over moves in root
    let mut nodes = 0;

    for m in root.iterate_moves() {
        assert!(root.make_move(m));
        nodes += perft_sub(root, depth - 1);
        root.unmake_move();
    }

    nodes
}

#[cfg(test)]
mod tests {
    use super::perft;
    use crate::position::Position;

    #[test]
    fn perft_default_fen_nochange() {
        println!("Running fen nochange test.");

        let mut p = Position::new();
        let starting_fen = p.get_fen();

        perft(&mut p, 3);

        assert_eq!(starting_fen, p.get_fen());
    }
}

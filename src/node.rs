use chess::{ChessMove, Color};

/// Node terminal cache enum.
pub enum TerminalStatus {
    Unknown,
    NotTerminal,
    Terminal(f32),
}

pub struct Node {
    pub p: f64,
    pub n: u32,
    pub tn: u32,
    pub w: f32,
    pub p_total: f64,
    pub maxdepth: usize,
    pub children: Option<Vec<usize>>,
    pub parent: Option<usize>,
    pub action: Option<ChessMove>,
    pub claim: bool,
    pub terminal: TerminalStatus,
    pub color: Color,
}

impl Node {
    /**
     * Creates a new tree with a single root node at id 0.
     */
    pub fn root(pov: Color) -> Self {
        Node {
            p: 0.0,
            p_total: 0.0,
            n: 0,
            tn: 0,
            w: 0.0,
            children: None,
            claim: false,
            parent: None,
            action: None,
            terminal: TerminalStatus::Unknown,
            color: pov,
            maxdepth: 0,
        }
    }

    /**
     * Constructs a new child node.
     */
    pub fn child(parent: usize, policy: f64, action: ChessMove, color: Color) -> Self {
        Node {
            p: policy,
            p_total: 0.0,
            n: 0,
            tn: 0,
            w: 0.0,
            children: None,
            claim: false,
            parent: Some(parent),
            action: Some(action),
            terminal: TerminalStatus::Unknown,
            color: color,
            maxdepth: 0,
        }
    }

    /// Gets the Q-value at this node. (average node score)
    pub fn q(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.w as f64 / self.n as f64
        }
    }
}

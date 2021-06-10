use chess::ChessMove;

/// Node terminal cache enum.
pub enum TerminalStatus {
    Unknown,
    NotTerminal,
    Terminal(f32),
}

pub struct Node {
    pub p: f32,
    pub n: u32,
    pub w: f32,
    pub p_total: f32,
    pub children: Option<Vec<usize>>,
    pub parent: Option<usize>,
    pub action: Option<chess::ChessMove>,
    pub claim: bool,
    pub terminal: TerminalStatus,
}

impl Node {
    /**
     * Creates a new tree with a single root node at id 0.
     */
    pub fn root() -> Self {
        Node {
            p: 0.0,
            p_total: 0.0,
            n: 0,
            w: 0.0,
            children: None,
            claim: false,
            parent: None,
            action: None,
            terminal: TerminalStatus::Unknown,
        }
    }

    /**
     * Constructs a new child node.
     */
    pub fn child(parent: usize, policy: f32, action: ChessMove) -> Self {
        Node {
            p: policy,
            p_total: 0.0,
            n: 0,
            w: 0.0,
            children: None,
            claim: false,
            parent: Some(parent),
            action: Some(action),
            terminal: TerminalStatus::Unknown,
        }
    }

    /// Gets the Q-value at this node. (average node score)
    pub fn q(&self) -> f32 {
        if self.n == 0 {
            0.0
        } else {
            self.w / self.n as f32
        }
    }
}

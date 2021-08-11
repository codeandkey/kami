use chess::{ChessMove, Color};
use serde::ser::{Serialize, Serializer, SerializeStruct};

/// Node terminal cache enum.
#[derive(Clone)]
pub enum TerminalStatus {
    Unknown,
    NotTerminal,
    Terminal(f64),
}

#[derive(Clone)]
pub struct Node {
    pub p: f64,
    pub n: u32,
    pub tn: u32,
    pub w: f64,
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

impl Serialize for Node {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("node", 7)?;

        state.serialize_field("n", &self.n)?;
        state.serialize_field("p", &self.p)?;
        state.serialize_field("w", &self.w)?;
        state.serialize_field("q", &self.q())?;
        state.serialize_field("tn", &self.tn)?;
        state.serialize_field("depth", &self.maxdepth)?;

        if let Some(action) = &self.action {
            state.serialize_field("action", &action.to_string())?;
        } else {
            state.serialize_field("action", "none")?;
        }

        state.end()
    }
}
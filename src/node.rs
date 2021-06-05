use chess::ChessMove;
use serde::ser::{Serialize, SerializeStruct, Serializer};

pub struct Node {
    pub p: f32,
    pub n: u32,
    pub w: f32,
    pub p_total: f32,
    pub children: Option<Vec<usize>>,
    pub parent: Option<usize>,
    pub action: Option<chess::ChessMove>,
    pub claim: bool,
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
        }
    }
}

impl Serialize for Node {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("Node", 5)?;

        state.serialize_field("n", &self.n)?;
        state.serialize_field("w", &self.w)?;
        state.serialize_field("p", &self.p)?;

        if self.n > 0 {
            state.serialize_field("q", &(self.w / (self.n as f32)))?;
        } else {
            state.serialize_field("q", &0)?;
        }

        if let Some(mv) = self.action {
            state.serialize_field("action", &mv.to_string())?;
        } else {
            state.serialize_field("action", "0000")?;
        }

        state.end()
    }
}

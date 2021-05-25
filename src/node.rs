use chess::ChessMove;
use serde::ser::{Serialize, Serializer, SerializeStruct};

use std::sync::{
    Arc, RwLock,
    atomic::{
        AtomicBool,
        Ordering
    }
};

struct Value {
    pub n: u32,
    pub w: f32,
}

impl Value {
    /**
     * Creates a new instance of Value.
     */
    pub fn new() -> Self {
        Value {
            n: 0,
            w: 0.0,
        }
    }

    /**
     * Adds to the value and increments the visit counter.
     */
    pub fn add(&mut self, v: f32) {
        self.n += 1;
        self.w += v;
    }

    /**
     * Gets both struct fields as a tuple.
     */
    pub fn both(&self) -> (u32, f32) {
        (self.n, self.w)
    }
}

pub struct Node {
    p: f32,
    children: RwLock<Option<Vec<usize>>>,
    claim: AtomicBool,
    value: RwLock<Value>,
    parent: Option<usize>,
    action: Option<chess::ChessMove>,
}

impl Node {
    /**
     * Creates a new tree with a single root node at id 0.
     */
    pub fn root() -> Self {
        Node {
            p: 0.0,
            children: RwLock::new(None),
            claim: AtomicBool::from(false),
            value: RwLock::new(Value::new()),
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
            children: RwLock::new(None),
            claim: AtomicBool::from(false),
            value: RwLock::new(Value::new()),
            parent: Some(parent),
            action: Some(action),
        }
    }

    /**
     * Assigns the children for this node.
     */
    pub fn assign_children(&mut self, children: Vec<usize>) {
        self.children.write().unwrap().insert(children);
    }

    /**
     * Tests if the node is currently claimed.
     */
    pub fn is_claimed(&self) -> bool {
        self.claim.load(Ordering::Relaxed)
    }

    /**
     * Tries to claim the node. Returns true if successfully claimed, false otherwise.
     */
    pub fn claim(&mut self) -> bool {
        self.claim.compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed).is_ok()
    }

    /**
     * Unclaims the node.
     */
    pub fn unclaim(&mut self) {
        self.claim.store(false, Ordering::Relaxed);
    }

    /**
     * Tests if the node has children.
     */
    pub fn has_children(&self) -> bool {
        self.children.read().unwrap().is_some()
    }

    /**
     * Gets the action at this node.
     */
    pub fn action(&self) -> Option<ChessMove> {
        self.action
    }

    /**
     * Returns a reference to this node's children.
     */
    pub fn get_children(&self) -> &Vec<usize> {
        &self.children.read().unwrap().unwrap()
    }

    /**
     * Returns the node's value structure.
     */
    pub fn get_value(&self) -> &Value {
        &self.value.read().unwrap()
    }

    /**
     * Gets this node's parent, if there is one.
     */
    pub fn get_parent(&self) -> Option<usize> {
        self.parent
    }

    /**
     * Adds to the node's value.
     */
    pub fn add(&mut self, value: f32) {
        self.value.write().unwrap().add(value);
    }
}

impl Serialize for Node {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where 
        S: Serializer
    {
        let mut state = serializer.serialize_struct("Node", 5)?;

        let (n, w) = self.value.read().unwrap().both();

        state.serialize_field("n", &n)?;
        state.serialize_field("w", &w)?;
        state.serialize_field("p", &self.p)?;

        if n > 0 {
            state.serialize_field("q", &(w / (n as f32)))?;
        } else {
            state.serialize_field("q", &0)?;
        }

        if let Some(mv) = self.action {
            state.serialize_field("action", &mv.to_string())?;
        } else {
            state.serialize_field("action", "none")?;
        }

        state.end()
    }
}
# Global immutable constants.

# Network architecture.

FRAME_COUNT = 6  # number of history frames
FRAME_SIZE  = 14 # history frame length
HEADER_SIZE = 18 # header length
RESIDUALS   = 4  # residual layer count
FILTERS     = 64 # filters per conv2d layer

# Network training parameters

LEARNING_RATE  = 1e-4 # optimizer learning rate (or initial LR)
DROPOUT        = 0.3  # dropout constant
POLICY_EPSILON = 1e-6 # policy epsilon to avoid NaN loss
EPOCHS         = 10   # training epoch count
L2_REG_WEIGHT  = 0.01 # L2 regularization weight

# Trainer parameters

NUM_SELFPLAY_GAMES = 10
NUM_ARENA_GAMES = 10
ARENACOMPARE_THRESHOLD = 0.6
NUM_TRAINING_BATCHES = 16
TRAINING_BATCH_SIZE = 16

# Tree search parameters

PUCT_NOISE_WEIGHT  = 0.25  # PUCT noise component weight
PUCT_POLICY_WEIGHT = 4.0   # PUCT policy component weight (prior + noise)
PUCT_NOISE_ALPHA   = 0.285 # PUCT dirichlet noise alpha
TREE_TEMPERATURE   = 1.0   # Move selection temperature
BATCH_SIZE         = 16

# Search control parameters

WORKER_PORT   = 8124   # Port to communciate with workers on
SEARCH_NODES  = 2500   # Nodes per search

MAX_RETRIES = 10
RETRY_DELAY = 1

# Webserver parameters

WEB_PORT = 8000
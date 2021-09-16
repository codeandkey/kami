# Global immutable constants

# Network architecture

FRAME_COUNT = 6  # number of history frames
FRAME_SIZE  = 14 # history frame length
HEADER_SIZE = 18 # header length
RESIDUALS   = 4  # residual layer count
FILTERS     = 16 # filters per conv2d layer

# Network training parameters

LEARNING_RATE  = 2e-4 # optimizer learning rate (or initial LR)
DROPOUT        = 0.3  # dropout constant
POLICY_EPSILON = 1e-6 # policy epsilon to avoid NaN loss
EPOCHS         = 10   # training epoch count
L2_REG_WEIGHT  = 0.01 # L2 regularization weight

# Trainer parameters

NUM_SELFPLAY_GAMES     = 4  # number of selfplay games (training window size)
NUM_ARENA_GAMES        = 0   # number of arenacompare games
SELFPLAY_WINDOW_SHIFT  = 4   # additional games to play after candidate rejected
ARENACOMPARE_THRESHOLD = 0.6 # min required win rate to accept candidate
NUM_TRAINING_BATCHES   = 8   # number of training batches
TRAINING_BATCH_SIZE    = 4   # positions per training batch
TRAINING_BUFFER_SIZE   = 16  # past games to train over

# Tree search parameters

PUCT_NOISE_WEIGHT         = 0.05  # PUCT noise component weight
PUCT_POLICY_WEIGHT        = 3     # PUCT policy component weight (prior + noise)
PUCT_NOISE_ALPHA          = 0.285 # PUCT dirichlet noise alpha
TREE_TEMPERATURE          = 1.0   # Move selection temperature
TREE_TEMPERATURE_DROP     = 1.0   # Move temperature (after drop ply)
TREE_TEMPERATURE_DROP_PLY = 30    # After nth ply, switch to drop temperature
BATCH_SIZE                = 16    # Maximum nodes per batch
ROLLOUT_WEIGHT            = 0     # Weight of classical MCTS rollout values in backprop (if zero, full rollouts disabled)

# Search control parameters

WORKER_PORT   = 8124   # Port to communciate with workers on
SEARCH_NODES  = 1200   # Nodes per search
MAX_RETRIES = 10
RETRY_DELAY = 1

# Webserver parameters

WEBSERVER_BIND = ('0.0.0.0', 8000) # Web server bind address and port (ipv4)

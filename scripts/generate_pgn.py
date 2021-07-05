import appdirs
import chess
import chess.pgn
import datetime
import json
import random
from os import path
import sys

# Locate archive directory
archive_dir = path.join(appdirs.user_data_dir(roaming=True), 'kami', 'archive')

if not path.exists(archive_dir):
    print('Archive directory "{}" not found'.format(archive_dir))
    sys.exit(-1)

max_generation = None
current_gen = 0

while path.exists(path.join(archive_dir, 'generation_{}'.format(current_gen))):
    max_generation = current_gen
    current_gen += 1

if max_generation is None:
    print('No generations have been archived')
    sys.exit(1)

# Count games in first generation
max_game = None
current_game = 0
while path.exists(path.join(archive_dir, 'generation_0', 'games', '{}.game'.format(current_game))):
    max_game = current_game
    current_game += 1

if max_game is None:
    print('No games have been archived (in first generation)')
    sys.exit(1)

def usage():
    print('Usage: {} [generationID] <gameNumber>')

    print('Available generations: 0-{}'.format(max_generation))
    print('Game numbers: 0-{}'.format(max_game))

# Check input generation
generation = int(sys.argv[1]) if len(sys.argv) > 1 else random.randint(0, max_generation)

if generation is None:
    usage()
    sys.exit(1)

if generation < 0 or generation > max_generation:
    print('Requested generation not available, provide a number between 0 and {} (inclusive)'.format(max_generation))
    sys.exit(1)

# Check input game
game = int(sys.argv[2]) if len(sys.argv) > 2 else random.randint(0, max_game)

if game is None:
    usage()
    sys.exit(1)

if game < 0 or game > max_game:
    print('Requested game not available, provide a number between 0 and {} (inclusive)'.format(max_game))
    sys.exit(1)

# Parse game data from disk
moves = None
mcts_values = None

game_path = path.join(archive_dir, 'generation_{}'.format(generation), 'games', '{}.game'.format(game))

with open(game_path, 'r') as f:
    game_data = json.load(f)
    moves = game_data['actions']
    mcts_values = game_data['mcts']

if moves is None or mcts_values is None:
    print('Game data at "{}" is corrupt, missing fields'.format(game_path))
    sys.exit(1)

# Generate game tree
pgn_game = chess.pgn.Game()

pgn_game.headers['Event'] = 'Kami Training Set / Generation {}'.format(generation)
pgn_game.headers['White'] = 'Kami Generation {}'.format(generation)
pgn_game.headers['Black'] = 'Kami Generation {}'.format(generation)
pgn_game.headers['Date'] = datetime.datetime.fromtimestamp(path.getctime(game_path))
pgn_game.headers['Round'] = str(game)

node = pgn_game

for i, mv in enumerate(moves):
    node = node.add_variation(chess.Move.from_uci(mv))

    # Find MCTS value for this move.
    value = None

    for move_value in mcts_values[i]:
        if move_value[0] == mv:
            value = move_value[1]

    if value is None:
        print('Failed to find move value for move {}:{}'.format(i, mv))
        sys.exit(1)

    node.comment = '{}%'.format(int(value * 100))

pgn_game.headers['Result'] = node.board().result(claim_draw=True)

print(pgn_game)
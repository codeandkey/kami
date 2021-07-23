# Displays live updates on network statistics.

#from scripts.train_torch import loss
import appdirs
import json
import matplotlib.pyplot as plt
import numpy as np
import time
from os import path
import sys

# Locate archive directory
archive_dir = path.join(appdirs.user_data_dir(roaming=True), 'kami', 'archive')

if not path.exists(archive_dir):
    print('Archive directory "{}" not found'.format(archive_dir))
    sys.exit(-1)

# Yields available generation paths in order.
def generations():
    current_gen = 0

    while True:
        nextpath = path.join(archive_dir, 'generation_{}'.format(current_gen))

        if not path.exists(nextpath):
            break

        yield nextpath
        current_gen += 1

# Yields games in JSON format from a generation in order.
def games(gen):
    current_game = 0

    while True:
        nextpath = path.join(gen, 'games', '{}.game'.format(current_game))

        if not path.exists(nextpath):
            break

        with open(nextpath) as f:
            yield json.load(f)

        current_game += 1

# Walks through generations and returns data for display.
# Returns (mean game ply / generation), elo_y, elo_x, data_loss_difference, data_loss_before_training, data_loss_after_training
def generate_datasets():
    p_loss_diff = []
    p_loss_pre = []
    p_loss_post = []
    v_loss_diff = []
    v_loss_pre = []
    v_loss_post = []
    loss_x = []
    elo_x = []
    elo_y = []

    for gen, genpath in enumerate(generations()):
        n = 0

        for game in games(genpath):
            n += 1
        
        if n == 0:
            break

        elo_path = path.join(genpath, 'games', 'elo')
        if path.exists(elo_path):
            elo_x.append(gen)

            with open(elo_path) as f:
                elo_y.append(int(f.read()))

        loss_path = path.join(genpath, 'loss')

        if path.exists(loss_path):
            with open(path.join(genpath, 'loss')) as f:
                parts = f.read().split(' ')
                
                p_loss_diff.append(float(parts[0]) - float(parts[1]))
                v_loss_diff.append(float(parts[2]) - float(parts[3]))
                p_loss_pre.append(float(parts[0]))
                v_loss_pre.append(float(parts[2]))
                p_loss_post.append(float(parts[1]))
                v_loss_post.append(float(parts[3]))
                loss_x.append(gen)

    return elo_x, elo_y, p_loss_pre, p_loss_post, p_loss_diff, v_loss_pre, v_loss_post, v_loss_diff, loss_x

# Collects data and renders it on the screen.
def render_plots():
    fig, ((elo, v_loss), (loss_diff, p_loss)) = plt.subplots(2, 2)
    fig.suptitle('Kami evolution')

    # Render average game ply
    elo_x, elo_y, p_loss_pre, p_loss_post, p_loss_diff, v_loss_pre, v_loss_post, v_loss_diff, loss_x = generate_datasets()

    elo.plot(elo_x, elo_y, 'b.-')
    elo.set_xlabel('Generation #')
    elo.set_ylabel('ELO projection')
    elo.set(ylim=(-20, max(elo_y) + 200), xlim=(0, max(elo_x)))
    elo.grid()

    loss_diff.plot(loss_x, p_loss_diff, 'b.-', label='policy')
    loss_diff.plot(loss_x, v_loss_diff, 'r.-', label='value')
    loss_diff.set_xlabel('Generation #')
    loss_diff.set_ylabel('Loss Δ')
    loss_diff.set(ylim=(0, max(max(p_loss_diff), max(v_loss_diff)) * 1.25), xlim=(0, len(loss_x) - 1))
    loss_diff.grid()
    loss_diff.legend()

    v_loss.plot(loss_x, v_loss_pre, 'r.-', label='value (pre)')
    v_loss.plot(loss_x, v_loss_post, 'y.-', label='value (post)')
    v_loss.set_xlabel('Generation #')
    v_loss.set_ylabel('Loss')
    v_loss.set(ylim=(0, max(max(v_loss_pre), max(v_loss_post)) * 1.25), xlim=(0, len(loss_x) - 1))
    v_loss.grid()
    v_loss.legend()

    p_loss.plot(loss_x, p_loss_pre, 'c.-', label='policy (pre)')
    p_loss.plot(loss_x, p_loss_post, 'b.-', label='policy (post)')
    p_loss.set_xlabel('Generation #')
    p_loss.set_ylabel('Loss')
    p_loss.set(ylim=(min(min(p_loss_pre), min(p_loss_post)) * 0.75, max(max(p_loss_pre), max(p_loss_post)) * 1.25), xlim=(0, len(loss_x) - 1))
    p_loss.grid()
    p_loss.legend()

    plt.show()

# Start rendering data.
while True:
    render_plots()
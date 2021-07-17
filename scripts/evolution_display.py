# Displays live updates on network statistics.

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
# Returns (mean game ply / generation), elo_y, elo_x, data_loss_difference, data_loss_after_training
def generate_datasets():
    data_mean = []
    data_loss_diff = []
    data_loss_after = []
    data_loss_x = []
    elo_x = []
    elo_y = []

    for gen, genpath in enumerate(generations()):
        total = 0
        n = 0

        for game in games(genpath):
            total += len(game['actions'])
            n += 1
        
        if n == 0:
            break

        elo_path = path.join(genpath, 'games', 'elo')
        if path.exists(elo_path):
            elo_x.append(gen)

            with open(elo_path) as f:
                elo_y.append(int(f.read()))

        data_mean.append(total / n)

        loss_path = path.join(genpath, 'loss')

        if path.exists(loss_path):
            with open(path.join(genpath, 'loss')) as f:
                parts = f.read().split(' ')
                
                data_loss_diff.append(float(parts[0]) - float(parts[1]))
                data_loss_after.append(float(parts[1]))
                data_loss_x.append(gen)

    return data_mean, elo_x, elo_y, data_loss_diff, data_loss_after, data_loss_x

# Collects data and renders it on the screen.
def render_plots():
    fig, ((elo, ply_mean), (loss_diff, loss_after)) = plt.subplots(2, 2)
    fig.suptitle('Kami model evolution statistics')

    # Render average game ply
    ply_mean_y, elo_x, elo_y, d_loss_diff, d_loss_after, d_loss_x = generate_datasets()
    ply_x = list(range(len(ply_mean_y)))

    ply_mean.plot(ply_x, ply_mean_y, '.-')
    ply_mean.set_xlabel('Generation #')
    ply_mean.set_ylabel('Average game ply')
    ply_mean.set(ylim=(0, 400))
    ply_mean.grid()

    elo.plot(elo_x, elo_y, '.-')
    elo.set_xlabel('Generation #')
    elo.set_ylabel('Estimated ELO rating')
    elo.set(ylim=(-20, max(elo_y) + 200), xlim=(0, len(ply_x) - 1))
    elo.grid()

    loss_diff.plot(d_loss_x, d_loss_diff, '.-')
    loss_diff.set_xlabel('Generation #')
    loss_diff.set_ylabel('Loss decrease')
    loss_diff.set(ylim=(0, max(d_loss_diff) * 1.25), xlim=(0, len(ply_x) - 1))
    loss_diff.grid()

    loss_after.plot(d_loss_x, d_loss_after, '.-')
    loss_after.set_xlabel('Generation #')
    loss_after.set_ylabel('Actual loss, post-train')
    loss_after.set(ylim=(0, max(d_loss_after) * 1.25), xlim=(0, len(ply_x) - 1))
    loss_after.grid()

    plt.show()

UPDATE_INTERVAL = 60 # display update interval, in seconds

# Start rendering data.
while True:
    render_plots()
    time.sleep(UPDATE_INTERVAL)
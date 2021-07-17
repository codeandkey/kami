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
# Returns (mean game ply / generation), elo_y, elo_x
def generate_datasets():
    data_mean = []
    elo_x = []
    elo_y = []

    for gen, genpath in enumerate(generations()):
        total = 0
        n = 0

        for game in games(genpath):
            total += len(game['actions'])
            n += 1

        elo_path = path.join(genpath, 'games', 'elo')
        if path.exists(elo_path):
            elo_x.append(gen)

            with open(elo_path) as f:
                elo_y.append(int(f.read()))

        data_mean.append(total / n)

    return data_mean, elo_x, elo_y

# Collects data and renders it on the screen.
def render_plots():
    fig, (elo, ply_mean) = plt.subplots(2, 1)
    fig.suptitle('Kami model evolution statistics')

    # Render average game ply
    ply_mean_y, elo_x, elo_y = generate_datasets()
    ply_x = list(range(len(ply_mean_y)))

    ply_mean.plot(ply_x, ply_mean_y, '.-')
    ply_mean.set_xlabel('Generation #')
    ply_mean.set_ylabel('Average game ply')
    ply_mean.set(ylim=(0, 400))
    ply_mean.grid()

    elo.plot(elo_x, elo_y, '.-')
    elo.set_xlabel('Generation #')
    elo.set_ylabel('Estimated ELO rating')
    elo.set(ylim=(0, 2500), xlim=(0, len(ply_mean_y)))
    elo.grid()

    plt.show()

UPDATE_INTERVAL = 60 # display update interval, in seconds

# Start rendering data.
while True:
    render_plots()
    time.sleep(UPDATE_INTERVAL)
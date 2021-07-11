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
# Returns (mean game ply / generation), (total ply / generation)
def generate_datasets():
    print('Computing average ply .. ', end='')
    data_mean = []
    data_total = []

    for genpath in generations():
        total = 0
        n = 0

        for game in games(genpath):
            total += len(game['actions'])
            n += 1

        data_mean.append(total / n)
        data_total.append(total)
    print('done')
    return data_mean, data_total

# Collects data and renders it on the screen.
def render_plots():
    fig, (ply_total, ply_mean) = plt.subplots(2, 1)
    fig.suptitle('Kami model evolution statistics')

    # Render average game ply
    ply_mean_y, ply_total_y = generate_datasets()
    ply_x = list(range(len(ply_mean_y)))

    ply_mean.plot(ply_x, ply_mean_y, '.-')
    ply_mean.set_xlabel('Generation #')
    ply_mean.set_ylabel('Average game ply')
    ply_mean.set(ylim=(0, 400))
    ply_mean.grid()

    ply_total.plot(ply_x, ply_total_y, '.-')
    ply_total.set(ylim=(0, 2500))
    ply_total.set_xlabel('Generation #')
    ply_total.set_ylabel('Total training set ply')
    ply_total.grid()

    plt.show()

UPDATE_INTERVAL = 60 # display update interval, in seconds

# Start rendering data.
while True:
    render_plots()
    time.sleep(UPDATE_INTERVAL)
# Game buffer type.

import json
from json.decoder import JSONDecodeError
from os import path
import os
import random

datapath = path.join(path.dirname(path.dirname(__file__)), 'data')

class Buffer:
    def __init__(self, name, size):
        """Initializes the buffer object."""
        self.name = name
        self.size = size

        # Initialize path
        self.path = path.join(datapath, '%s.json' % name)

        # Try and load buffer data
        try:
            with open(self.path, 'r') as f:
                self.games = json.load(f)
        except FileNotFoundError:
            self.games = []

    def add(self, game):
        """Adds a single item to the buffer. Exports the buffer to the disk."""
        self.games.append(game)

        while len(self.games) > self.size:
            self.games.pop(0)

        # Try and write buffer data
        with open(self.path, 'w') as f:
            json.dump(self.games, f)

    def choose(self):
        return random.choice(self.games)

# Model training loop

import consts
from model import Model
from search import Search

import chess
import json
import os
from os import path
import random
import threading

datapath = path.join(path.dirname(path.dirname(__file__)), 'data')

candidate_path = path.join(datapath, 'candidate.pt')
model_path = path.join(datapath, 'model.pt')
selfplay_games_path = path.join(datapath, 'selfplay')
arena_games_path = path.join(datapath, 'arena')

class Trainer:
    def __init__(self):
        os.makedirs(datapath, exist_ok=True)
        os.makedirs(selfplay_games_path, exist_ok=True)
        os.makedirs(arena_games_path, exist_ok=True)

        self.status = {}

    def start_training(self):
        """Starts the training process on a separate thread."""
        self.thread = threading.Thread(target=self.trainer_thread_main)
        self.thread.start()

    def trainer_thread_main(self):
        # Generate initial model if needed
        if not path.exists(model_path):
            print('Generating new model at %s' % model_path)
            Model().save(model_path)

        # Generate selfplay set
        self.gen_selfplay_games()

        # Generate candidates and do arenacompare until a nextgen is found
        while True:
            self.generate_candidate()
            self.gen_arenacompare_games()

            if self.get_arenacompare_winrate() >= consts.ARENACOMPARE_THRESHOLD:
                os.remove(model_path)
                os.rename(candidate_path, model_path)
            else:
                os.remove(candidate_path)

    def update_search_status(self, stat):
        self.status['search'] = stat

    def gen_selfplay_games(self):
        needed_games = []

        # Generate selfplay set
        for i in range(consts.NUM_SELFPLAY_GAMES):
            gpath = path.join(selfplay_games_path, '{}.json'.format(i))

            if path.exists(gpath):
                continue

            # Generate selfplay game
            needed_games.append(gpath)
        
        if len(needed_games) > 0:
            print('Generating %s selfplay games' % len(needed_games))
            s = Search(model_path)

            self.status['task'] = 'selfplay'

            for target in needed_games:
                game = self.play_game(s, s)

                with open(target, 'w') as f:
                    f.write(json.dumps(game))

                print('Finished selfplay game {}, result {}'.format(target, game['result']))

            s.stop()

    def play_game(self, white, black):
        current_game = {
            'steps': []
        }

        pos = chess.Board()

        white.reset()
        black.reset()

        while not pos.is_game_over(claim_draw=True):
            result = None

            if pos.turn == chess.WHITE:
                result = white.go(lambda stat: self.update_search_status(stat))
            else:
                result = black.go(lambda stat: self.update_search_status(stat))

            white.push(result['action'])

            if black != white:
                black.push(result['action'])

            # Select move
            pos.push(chess.Move.from_uci(result['action']))
            current_game['steps'].append(result)

        outcome = pos.outcome(claim_draw=True)

        if outcome.result() == '1/2-1/2':
            current_game['result'] = 0
        elif outcome.result() == '1-0':
            current_game['result'] = 1
        else:
            current_game['result'] = -1

        return current_game
    
    def gen_arenacompare_games(self):
        needed_games = []

        # Generate selfplay set
        for i in range(consts.NUM_ARENA_GAMES):
            gpath = path.join(arena_games_path, '{}.json'.format(i))

            if path.exists(gpath):
                continue

            # Generate selfplay game
            needed_games.append(gpath)
        
        if len(needed_games) > 0:
            print('Generating {} arenacompare games' % len(needed_games))
            current = Search(model_path)
            candidate = Search(candidate_path)

            self.status['task'] = 'arenacompare'

            for target in needed_games:
                if random.randint(0, 1) == 0:
                    white = candidate
                    black = current
                    cside = 1
                else:
                    white = current
                    black = candidate
                    cside = -1

                game = self.play_game(white, black)
                game['cside'] = cside

                with open(target, 'w') as f:
                    f.write(json.dumps(game))

                print('Finished arenacompare game {}, result {} (for candidate)'.format(target, game['cside'] * game['result']))
            
            current.stop()
            candidate.stop()
    
    def generate_candidate(self):
        """Generates a candidate model from selfplay data."""

        # If a candidate exists, skip
        if path.exists(candidate_path):
            return

        for i in range(consts.NUM_ARENA_GAMES):
            p = path.join(arena_games_path, '{}.json'.format(i))

            if path.exists(p):
                os.remove(p)

        # First generate n training batches.
        tbatches = [self.generate_training_batch() for _ in range(consts.NUM_TRAINING_BATCHES)]

        new_candidate = Model(model_path)
        new_candidate.train(tbatches)
        new_candidate.save(candidate_path)

    def generate_training_batch(self):
        """Generates a single training batch from selfplay data."""
        outbatch = []

        for _ in range(consts.TRAINING_BATCH_SIZE):
            gsel = random.randint(0, consts.NUM_SELFPLAY_GAMES - 1)
            gdata = json.loads(open(path.join(selfplay_games_path, '{}.json'.format(gsel))).read().decode('utf-8'))

            fsel = random.randint(0, len(gdata['steps']) - 1)

            result_mul = 1

            if fsel % 2 == 1:
                result_mul = -1
            
            outbatch.append(
                (gdata['steps'][fsel]['headers'], gdata['steps'][fsel]['frames'], gdata['steps'][fsel]['lmm']),
                (gdata['steps'][fsel]['mcts'], gdata['result'] * result_mul)
            )

        return outbatch
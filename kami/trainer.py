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
gen_path = path.join(datapath, 'generation')
archive_path = path.join(datapath, 'archive')

class Trainer:
    def __init__(self):
        os.makedirs(datapath, exist_ok=True)
        os.makedirs(selfplay_games_path, exist_ok=True)
        os.makedirs(arena_games_path, exist_ok=True)
        os.makedirs(archive_path, exist_ok=True)

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

        if not path.exists(gen_path):
            print('Marked generation 0')
            with open(gen_path, 'w') as f:
                f.write('0')

        # Generate candidates and do arenacompare until a nextgen is found
        while True:
            self.generate_candidate()

            print('Starting arenacompare phase')
            accept = self.do_arenacompare()

            if accept:
                print('Accepting candidate')
                self.advance_candidate()
            else:
                print('Rejecting candidate')

                os.remove(candidate_path)

                # Remove the oldest n selfplay games.

                for i in range(consts.SELFPLAY_WINDOW_SHIFT):
                    p = path.join(selfplay_games_path, '{}.json'.format(i))
                    
                    if path.exists(p):
                        os.remove(p)

                # Shift the existing selfplay games down.

                for i in range(consts.SELFPLAY_WINDOW_SHIFT, consts.NUM_SELFPLAY_GAMES):
                    src = path.join(selfplay_games_path, '{}.json'.format(i))
                    dst = path.join(selfplay_games_path, '{}.json'.format(i - consts.SELFPLAY_WINDOW_SHIFT))

                    os.rename(src, dst)

    def update_search_status(self, stat):
        self.status['tree'] = stat['tree']
        self.status['fen'] = stat['fen']

        total_n = sum(map(lambda nd: nd['n'], stat['tree']))
        self.status['progress'] = total_n / consts.SEARCH_NODES

        self.status['nps'] = total_n / (stat['elapsed'] / 1000)

    def advance_candidate(self):
        """Accepts the current candidate as the next generation."""
        os.remove(model_path)
        os.rename(candidate_path, model_path)
    
        gen = int(open(gen_path, 'r').read())

        # Move selfplay games to archive
        target_archive_dir = path.join(archive_path, 'generation_%s' % gen)
        os.makedirs(target_archive_dir)

        for i in range(consts.NUM_SELFPLAY_GAMES):
            p = path.join(selfplay_games_path, '{}.json'.format(i))
            
            if path.exists(p):
                os.rename(p, path.join(target_archive_dir, '{}.json'.format(i)))

        # Flush arenacompare games
        for i in range(consts.NUM_ARENA_GAMES):
            p = path.join(arena_games_path, '{}.json'.format(i))

            if path.exists(p):
                os.remove(p)

        # Write generation marker
        with open(gen_path, 'w') as f:
            f.write(str(gen + 1))

        print('Accepted candidate as generation {}'.format(gen + 1))

    def do_selfplay(self):
        """Generates all required selfplay games."""
        s = None

        # Generate selfplay set
        for i in range(consts.NUM_SELFPLAY_GAMES):
            gpath = path.join(selfplay_games_path, '{}.json'.format(i))

            if path.exists(gpath):
                continue

            print('Generating selfplay game {} of {}'.format(i + 1, consts.NUM_SELFPLAY_GAMES))

            self.status['state'] = 'Selfplay: generation {}, {} of {}'.format(
                open(gen_path).read(),
                i + 1,
                consts.NUM_SELFPLAY_GAMES,
            )

            if s is None:
                s = Search(model_path)

            self.status['task'] = 'selfplay'

            game = self.play_game(s, s)

            with open(gpath, 'w') as f:
                f.write(json.dumps(game))

            print('Finished game, result {}'.format(game['result']))

        if s:
            s.stop()

    def play_game(self, white, black):
        """Plays a single game between one or more searchers."""
        current_game = {
            'steps': []
        }

        self.status['actions'] = []
        self.status['depth'] = []
        self.status['score']= []

        pos = chess.Board()

        white.reset()
        black.reset()

        while not pos.is_game_over(claim_draw=True):
            result = None

            if pos.turn == chess.WHITE:
                result = white.go(lambda stat: self.update_search_status(stat))
            else:
                result = black.go(lambda stat: self.update_search_status(stat))

            if 'result' in result:
                self.status['fen'] = result['fen']
                current_game['result'] = result['result']
                break

            white.push(result['action'])

            if black != white:
                black.push(result['action'])

            current_game['steps'].append(result)

            self.status['depth'].append(result['depth'])
            self.status['actions'].append(result['action'])
            self.status['score'].append(result['score'])

        return current_game
    
    def do_arenacompare(self):
        """Performs the arenacompare phase. Returns true if the candidate
           should be accepted, false otherwise."""

        current = None
        candidate = None

        score = 0
        accept = None

        # Generate selfplay set
        for i in range(consts.NUM_ARENA_GAMES):
            gpath = path.join(arena_games_path, '{}.json'.format(i))

            if path.exists(gpath):
                result = json.loads(open(gpath, 'r').read())
                score += ((result['cside'] * result['result']) + 1) / 2
                continue

            if current is None:
                current = Search(model_path)
            
            if candidate is None:
                candidate = Search(candidate_path, consts.WORKER_PORT + 1)

            print('Generating arenacompare game {} of {}'.format(i + 1, consts.NUM_ARENA_GAMES))

            if random.randint(0, 1) == 0:
                white = candidate
                black = current
                cside = 1
                print('White: Candidate, Black: Current')
            else:
                white = current
                black = candidate
                cside = -1
                print('White: Current, Black: Candidate')

            self.status['state'] = 'Arena: {} vs. {} ({} of {}, WR {})'.format(
                open(gen_path).read() if cside == -1 else 'candidate',
                open(gen_path).read() if cside == 1 else 'candidate',
                i + 1,
                consts.NUM_ARENA_GAMES,
                score / consts.NUM_ARENA_GAMES,
            )

            game = self.play_game(white, black)
            game['cside'] = cside

            with open(gpath, 'w') as f:
                f.write(json.dumps(game))

            score += ((game['cside'] * game['result']) + 1) / 2

            print('Finished arenacompare game {}, result {} (for candidate)'.format(gpath, game['cside'] * game['result']))

            # Check if candidate can accept or reject.

            best_possible_score = score + (consts.NUM_ARENA_GAMES - (i + 1))
            worst_possible_score = score

            if best_possible_score < consts.NUM_ARENA_GAMES * consts.ARENACOMPARE_THRESHOLD:
                print('Best possible winrate {} below threshold {}, rejecting'.format(
                    best_possible_score / consts.NUM_ARENA_GAMES,
                    consts.ARENACOMPARE_THRESHOLD,
                ))
                accept = False
                break

            if worst_possible_score >= consts.NUM_ARENA_GAMES * consts.ARENACOMPARE_THRESHOLD:
                print('Worst possible winrate {} above threshold {}, accepting'.format(
                    worst_possible_score / consts.NUM_ARENA_GAMES,
                    consts.ARENACOMPARE_THRESHOLD,
                ))
                accept = True
                break

            print('Continuing games, winrate window [{} => {}] (target {})'.format(
                worst_possible_score / consts.NUM_ARENA_GAMES,
                best_possible_score / consts.NUM_ARENA_GAMES,
                consts.ARENACOMPARE_THRESHOLD
            ))
            
        if current:
            current.stop()
        if candidate:
            candidate.stop()
        
        return accept

    def generate_candidate(self):
        """Generates a candidate model from selfplay data."""

        # If a candidate exists, skip
        if path.exists(candidate_path):
            return

        # Generate training set
        self.do_selfplay()

        print('Generating candidate model')
        self.status['state'] = 'Generating candidate model'

        # Flush arenacompare games
        for i in range(consts.NUM_ARENA_GAMES):
            p = path.join(arena_games_path, '{}.json'.format(i))

            if path.exists(p):
                os.remove(p)

        # Start a searcher for batch gen.
        s = Search(model_path)

        # First generate n training batches.
        tpositions = [self.generate_training_batch_positions() for _ in range(consts.NUM_TRAINING_BATCHES)]
        tbatches = [s.make_batch(tpos) for tpos in tpositions]

        s.stop()

        new_candidate = Model(model_path)
        startloss, endloss = new_candidate.train(tbatches)

        print('Candidate average loss diff: {} => {}'.format(startloss, endloss))

        new_candidate.save(candidate_path)

    def generate_training_batch_positions(self):
        """Generates a list of selected training positions."""
        positions = []

        for _ in range(consts.TRAINING_BATCH_SIZE):
            gsel = random.randint(0, consts.NUM_SELFPLAY_GAMES - 1)
            gdata = json.loads(open(path.join(selfplay_games_path, '{}.json'.format(gsel))).read())

            fsel = random.randint(0, len(gdata['steps']) - 1)

            actions = []

            for i in range(fsel):
                actions.append(gdata['steps'][i]['action'])

            positions.append((actions, gdata['steps'][fsel]['mcts_pairs'], gdata['result']))

        return positions
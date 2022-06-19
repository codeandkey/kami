#pragma once

#include <atomic>
#include <iostream>

#include <vector>

#include <cstring>
#define strcpy_s(s, n, p) strncpy(s, p, n)
#include "chess/thc/thc.h"
#include "chess/neocortex/move.h"
#include "chess/neocortex/piece.h"
#include "chess/neocortex/position.h"

namespace kami {
constexpr int NFEATURES = 8 + 6 + 4 + 12;
constexpr int PSIZE = 64 * 64 + (8 + 14) * 4;
constexpr int WIDTH = 8;
constexpr int HEIGHT = 8;
constexpr int OBSIZE = WIDTH * HEIGHT * NFEATURES;

class NCInit {
    public: NCInit() {
        static std::atomic<bool> initialized = false;

        if (!initialized.exchange(true)) {
            neocortex::bb::init();
            neocortex::zobrist::init();
            neocortex::attacks::init();
            std::cout << "Initialized neocortex lookup tables" << std::endl;
        }
    }
};

static NCInit nc_initializer;

class Env {
    private:
        float curturn;
        std::vector<neocortex::Move> history;
        std::vector<int> cur_actions;
        bool actions_utd;

        neocortex::Position game;

    public:

        Env() {
            curturn = 1.0f;
            actions_utd = false;
        }

        int ply() { return history.size(); }

        int encode(neocortex::Move move) 
        {
            int base;

            switch (move.ptype())
            {
                case neocortex::piece::QUEEN:
                    base = 4096;
                    break;
                case neocortex::piece::ROOK:
                    base = 4096 + 22;
                    break;
                case neocortex::piece::BISHOP:
                    base = 4096 + (22 * 2);
                    break;
                case neocortex::piece::KNIGHT:
                    base = 4096 + (22 * 3);
                    break;
                case neocortex::piece::null:
                    return move.src() * 64 + move.dst();
            }

            // Push
            char srcfile, dstfile;

            srcfile = neocortex::square::file(move.src());
            dstfile = neocortex::square::file(move.dst());

            if (srcfile == dstfile)
                return base + srcfile;

            if (srcfile == dstfile + 1)
                return base + 8 + srcfile - 1;

            return base + 15 + srcfile;
        }

        neocortex::Move decode(int action)
        {
            if (action < 4096)
                return neocortex::Move(action / 64, action % 64);

            action -= 4096;

            int ptypes[] = {
                neocortex::piece::QUEEN,
                neocortex::piece::ROOK,
                neocortex::piece::KNIGHT,
                neocortex::piece::BISHOP,
            };

            int ptype = ptypes[action / 22];
            int srcrank, dstrank;
            int srcfile, dstfile;

            action %= 22;

            if (game.get_color_to_move() == neocortex::piece::WHITE)
            {
                srcrank = 6;
                dstrank = 7;
            } else
            {
                srcrank = 1;
                dstrank = 0;
            }

            if (action < 8)
            {
                srcfile = dstfile = action;
            } else
            {
                action -= 8;

                // Leftcap
                if (action < 7)
                {
                    srcfile = action + 1;
                    dstfile = action;
                } else
                {
                    // Rightcap
                    action -= 7;

                    srcfile = action;
                    dstfile = action + 1;
                }
            }

            return neocortex::Move(
                neocortex::square::at(srcrank, srcfile),
                neocortex::square::at(dstrank, dstfile),
                ptype
            );
        }

        void observe(float* dst) {
            float header[8 + 6 + 4];

            // Clear observation
            memset(dst, 0, sizeof(float) * OBSIZE);

            // Build square headers
            int ply = history.size();

            for (int i = 0; i < 8; ++i)
                header[i] = (ply >> i) & 1;

            int hmc = game.halfmove_clock();
            for (int i = 0; i < 6; ++i)
                header[8 + i] = (hmc >> i) & 1;

            header[14] = game.has_castle_right(neocortex::CASTLE_WHITE_K) ? 1.0f : 0.0f;
            header[15] = game.has_castle_right(neocortex::CASTLE_WHITE_Q) ? 1.0f : 0.0f;
            header[16] = game.has_castle_right(neocortex::CASTLE_BLACK_K) ? 1.0f : 0.0f;
            header[17] = game.has_castle_right(neocortex::CASTLE_BLACK_Q) ? 1.0f : 0.0f;

            // Write all square headers
            for (int sq = 0; sq < 64; ++sq)
                memcpy(dst + sq * NFEATURES, header, sizeof(header));

            bool our_col = curturn < 0;

            for (int rank = 0; rank < 8; ++rank)
            {
                for (int file = 0; file < 8; ++file)
                {
                    float* base = dst;

                    if (our_col)
                        base += (rank * WIDTH * NFEATURES) + (file * NFEATURES);
                    else
                        base += ((7 - rank) * WIDTH * NFEATURES) + ((7 - file) * NFEATURES);

                    // Note: board is flipped vertically due to THC but this should be OK anyway
                    int pc = game.get_board().get_piece(rank * 8 + file);

                    if (!neocortex::piece::is_valid(pc))
                        continue;

                    if (neocortex::piece::color(pc) == game.get_color_to_move())
                        base += 18;
                    else
                        base += 24;

                    base[neocortex::piece::type(pc)] = 1.0f;
                }
            }
        }

        void push(int action)
        {
            neocortex::Move mv = decode(action);
            game.make_move(mv);
            history.push_back(mv);
            curturn = -curturn;
            actions_utd = false;
        }

        void pop()
        {
            game.unmake_move(history.back());
            history.pop_back();
            curturn = -curturn;
            actions_utd = false;
        }

        std::string debug_action(int action)
        {
            return decode(action).to_uci();
        }

        bool terminal_str(float* value, std::string& out)
        {
            // 50-move rule
            if (game.halfmove_clock() >= 50)
            {
                *value = 0;
                out = "Draw by 50-move rule";
                return true;
            }

            // Threefold repetition (usually faster than movegen)
            if (game.num_repetitions() >= 3)
            {
                *value = 0;
                out = "Draw by threefold repetition";
                return true;
            }

            // Insufficient material
            neocortex::bitboard kings, knights, bishops, global, white, black;

            kings = game.get_board().get_piece_occ(neocortex::piece::KING);
            knights = game.get_board().get_piece_occ(neocortex::piece::KNIGHT);
            bishops = game.get_board().get_piece_occ(neocortex::piece::BISHOP);
            global = game.get_board().get_global_occ();
            white = game.get_board().get_color_occ(neocortex::piece::WHITE);
            black = game.get_board().get_color_occ(neocortex::piece::BLACK);

            if (
                kings == global // K vs K

                || (global == (kings | bishops) && ( // K/B endgame
                        neocortex::bb::popcount(bishops) == 1 // K vs KB
                        || (neocortex::bb::popcount(white) == neocortex::bb::popcount(black) && neocortex::bb::popcount(bishops) == 2) // KB vs KB
                   ))

                || (global == (kings | knights) && ( // K/N endgame
                        neocortex::bb::popcount(knights) == 1 // K vs KN
                        || (neocortex::bb::popcount(white) == neocortex::bb::popcount(black) && neocortex::bb::popcount(knights) == 2) // KN vs KN
                   ))
            ) 
            {
                *value = 0;
                out = "Draw by insufficient material";
                return true;
            }

            // Test if at least one legal move
            if (actions().size())
                return false;

            if (game.check())
            {
                if (game.get_color_to_move() == neocortex::piece::WHITE)
                {
                    *value = -1.0f;
                    out = "White is checkmated";
                } else
                {
                    *value = 1.0f;
                    out = "Black is checkmated";
                }

                // DEBUG crazy terminals
                if (history.size() <= 3) // impossible game
                {
                    neocortex::Move moves[neocortex::MAX_PL_MOVES];
                    int npl = game.pseudolegal_moves(moves);
                    std::cout << "actions.size(): " << actions().size() << std::endl;
                    std::cout << "actions:";
                    for (auto& i : actions())
                        std::cout << " " << i;
                    std::cout << "history:";
                    for (auto& i : history)
                        std::cout << " " << i.to_uci();
                    std::cout << std::endl;
                    std::cout << "state: " << game.to_fen() << std::endl;
                    std::cout << "check: " << game.check() << std::endl;
                    std::cout << "PL moves:";
                    for (int i = 0; i < npl; ++i)
                        std::cout << " " << moves[i].to_uci();
                    std::cout << std::endl;
                    throw std::runtime_error("impossible game occurred");
                }

                return true;
            }

            *value = 0.0f;

            if (game.get_color_to_move() == neocortex::piece::WHITE)
                out = "White is stalemated";
            else
                out = "Black is stalemated";

            return true;
        }

        bool terminal(float* value)
        {
            std::string unused;
            return terminal_str(value, unused);
        }

        float turn()
        {
            return curturn;
        }

        std::vector<int>& actions()
        {
            if (!actions_utd)
            {
                // Generate moves
                neocortex::Move moves[neocortex::MAX_PL_MOVES];
                int n = game.pseudolegal_moves(moves);
                game.order_moves(moves, n);

                cur_actions.clear();

                // Test if at least one legal move
                for (int i = 0; i < n; ++i)
                {
                    if (game.make_move(moves[i]))
                        cur_actions.push_back(encode(moves[i]));

                    game.unmake_move(moves[i]);
                }

                actions_utd = true;
            }

            return cur_actions;
        }

        std::string print()
        {
            return game.to_fen();
        }

        std::string pgn()
        {
            // Expect the game to be in a terminal position.
            float value;
            std::string tstr;
            if (!terminal_str(&value, tstr))
                throw std::runtime_error("Game must be in terminal state to write PGN!");

            std::string result, output;

            if (value < 0)
                result = "0-1";
            else if (value > 0)
                result = "1-0";
            else
                result = "1/2-1/2";

            result += " {" + tstr + "}";

            int mn = 1;

            thc::ChessRules board;
            board.Init();

            // Walk through move history.
            for (auto& mv : history)
            {
                if (board.WhiteToPlay())
                    output += (mn == 1 ? "" : " ") + std::to_string(mn) + ".";
                else
                    ++mn;

                thc::Move move;
                std::string uci = mv.to_uci();
                move.TerseIn(&board, uci.c_str());
                output += " " + move.NaturalOut(&board);

                board.PushMove(move);
            }

            return output + " " + result;
        }

        float bootstrap_value(float window)
        {
            float score = (float) game.evaluate() / window;
            
            score = std::min(score, 1.0f);
            score = std::max(score, -1.0f);

            return score;
        }
};
}

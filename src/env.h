#pragma once

#include <iostream>

#include <vector>

#include <cstring>
#define strcpy_s(s, n, p) strncpy(s, p, n)
#include "chess/thc.h"

namespace kami {
constexpr int NFEATURES = 8 + 6 + 4 + 12;
constexpr int PSIZE = 64 * 64 + (8 + 14) * 4;
constexpr int WIDTH = 8;
constexpr int HEIGHT = 8;
constexpr int OBSIZE = WIDTH * HEIGHT * NFEATURES;

class Env {
    private:
        float curturn;
        std::vector<thc::Move> history;
        std::vector<char*> square_hist;
        std::vector<int> cur_actions;
        bool actions_utd;

    public:
        thc::ChessRules board;
        std::vector<int> repetitions;
        std::vector<int> halfmove_clock;

        Env() {
            curturn = 1.0f;
            repetitions.push_back(1);
            halfmove_clock.push_back(0);

            char* squares_dup = new char[sizeof(board.squares)];
            memcpy(squares_dup, board.squares, sizeof board.squares);
            square_hist.push_back(squares_dup);
            actions_utd = false;
        }

        int ply() { return history.size(); }

        int encode(thc::Move move) 
        {
            int base;

            switch (move.special)
            {
                case thc::SPECIAL_PROMOTION_QUEEN:
                    base = 4096;
                    break;
                case thc::SPECIAL_PROMOTION_ROOK:
                    base = 4096 + 22;
                    break;
                case thc::SPECIAL_PROMOTION_BISHOP:
                    base = 4096 + (22 * 2);
                    break;
                case thc::SPECIAL_PROMOTION_KNIGHT:
                    base = 4096 + (22 * 3);
                    break;
                default:
                    return move.src * 64 + move.dst;
            }

            // Push
            char srcfile, dstfile;

            srcfile = move.src & 7;
            dstfile = move.dst & 7;

            if (srcfile == dstfile)
                return base + srcfile;

            if (srcfile == dstfile + 1)
                return base + 8 + srcfile - 1;

            return base + 15 + srcfile;
        }

        thc::Move decode(int action)
        {
            thc::Move out;
            out.Invalid();

            char mv[6] = {0};

            if (action < 4096)
            {
                thc::Square src = static_cast<thc::Square>(action / 64);
                thc::Square dst = static_cast<thc::Square>(action % 64);

                mv[0] = thc::get_file(src);
                mv[1] = thc::get_rank(src);
                mv[2] = thc::get_file(dst);
                mv[3] = thc::get_rank(dst);

                out.TerseIn(&board, mv);
                return out;
            }

            action -= 4096;

            const char* ptypes = "qrbn";
            mv[4] = ptypes[action / 22];
            action %= 22;

            mv[1] = board.WhiteToPlay() ? '7' : '2';
            mv[3] = board.WhiteToPlay() ? '8' : '1';

            if (action < 8)
            {
                mv[0] = mv[2] = 'a' + action;

                out.TerseIn(&board, mv);
                return out;
            }

            action -= 8;

            // Leftcap
            if (action < 7)
            {
                mv[0] = 'b' + action;
                mv[2] = 'a' + action;

                out.TerseIn(&board, mv);
                return out;
            }

            action -= 7;

            // Rightcap
            mv[0] = 'a' + action;
            mv[2] = 'b' + action;

            out.TerseIn(&board, mv);
            return out;
        }

        void observe(float* dst) {
            float header[8 + 6 + 4];

            // Clear observation
            for (int i = 0; i < OBSIZE; ++i)
                dst[i] = 0.0f;

            // Build square headers
            for (int i = 0; i < 8; ++i)
                header[i] = (history.size() >> (i + 1)) & 1;

            for (int i = 0; i < 6; ++i)
                header[8 + i] = (board.half_move_clock >> i) & 1;

            header[14] = board.wking_allowed() ? 1.0f : 0.0f;
            header[15] = board.wqueen_allowed() ? 1.0f : 0.0f;
            header[16] = board.bking_allowed() ? 1.0f : 0.0f;
            header[17] = board.bqueen_allowed() ? 1.0f : 0.0f;

            bool our_col = curturn < 0;

            for (int rank = 0; rank < 8; ++rank)
            {
                for (int file = 0; file < 8; ++file)
                {
                    float* base;

                    if (our_col)
                        base = dst + (rank * WIDTH * NFEATURES) + (file * NFEATURES);
                    else
                        base = dst + ((7 - rank) * WIDTH * NFEATURES) + ((7 - file) * NFEATURES);

                    for (int h = 0; h < 18; ++h)
                        base[h] = header[h];

                    char pc = board.squares[(7 - rank) * 8 + file];

                    base += 18;

                    bool pc_col = pc == tolower(pc);

                    if (pc_col != our_col)
                        base += 6;

                    pc = tolower(pc);

                    switch (pc)
                    {
                        case 'p':
                            base[0] = 1.0f;
                            break;
                        case 'b':
                            base[1] = 1.0f;
                            break;
                        case 'n':
                            base[2] = 1.0f;
                            break;
                        case 'r':
                            base[3] = 1.0f;
                            break;
                        case 'q':
                            base[4] = 1.0f;
                            break;
                        case 'k':
                            base[5] = 1.0f;
                            break;
                    }
                }
            }
        }

        void push(int action)
        {
            thc::Move mv = decode(action);
            //std::cout << "pushing " << mv.TerseOut() << std::endl;
            history.push_back(mv);
            board.PlayMove(history.back());
            curturn = -curturn;
            char* squares_dup = new char[sizeof(board.squares)];
            memcpy(squares_dup, board.squares, sizeof board.squares);
            square_hist.push_back(squares_dup);

            int hmc = halfmove_clock.back() + 1;

            if (mv.capture != ' ' || board.squares[mv.src] == 'p' || board.squares[mv.src] == 'P')
                hmc = 0;

            halfmove_clock.push_back(hmc);

            int reps = 0;

            for (unsigned i = 0; i < square_hist.size(); ++i)
                if (!memcmp(squares_dup, square_hist[i], sizeof board.squares))
                    reps++;

            repetitions.push_back(reps);
            actions_utd = false;
        }

        void pop()
        {
            //std::cout << "popping " << history.back().TerseOut() << std::endl;
            board.PopMove(history.back());
            history.pop_back();
            curturn = -curturn;

            delete[] square_hist.back();
            square_hist.pop_back();
            repetitions.pop_back();
            halfmove_clock.pop_back();
            actions_utd = false;
        }

        bool terminal_str(float* value, std::string& out)
        {
            thc::TERMINAL val;
            thc::DRAWTYPE drawtype;

            board.Evaluate(val);

            switch (val)
            {
                case thc::TERMINAL_BCHECKMATE:
                    *value = 1.0f;
                    out = "Black is checkmated";
                    return true;
                case thc::TERMINAL_WCHECKMATE:
                    *value = -1.0f;
                    out = "White is checkmated";
                    return true;
                case thc::TERMINAL_WSTALEMATE:
                    *value = 0.0f;
                    out = "White is stalemated";
                    return true;
                case thc::TERMINAL_BSTALEMATE:
                    *value = 0.0f;
                    out = "Black is stalemated";
                    return true;
                default:;
            }

            if (repetitions.back() >= 3)
            {
                out = "Draw by threefold repetition";
                *value = 0.0f;
                return true;
            }

            if (halfmove_clock.back() >= 100)
            {
                out = "Draw by 50-move rule";
                *value = 0.0f;
                return true;
            }

            if (board.IsInsufficientDraw(false, drawtype) && drawtype == thc::DRAWTYPE::DRAWTYPE_INSUFFICIENT_AUTO)
            {
                out = "Draw by insufficient material";
                *value = 0.0f;
                return true;
            }

            return false;
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
                std::vector<thc::Move> moves;
                cur_actions.clear();

                board.GenLegalMoveList(moves);

                for (auto& m : moves)
                    cur_actions.push_back(encode(m));

                actions_utd = true;
            }

            return cur_actions;
        }

        std::string print()
        {
            return board.ToDebugStr();
        }

        void lmm(float* out)
        {
            for (int i = 0; i < PSIZE; ++i)
                out[i] = 0.0f;

            for (int& i : actions())
                out[i] = 1.0f;
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

            std::vector<std::string> moves;
            std::vector<thc::Move> history_back;

            // Walk through move history.
            while (history.size())
            {
                thc::Move move = history.back();
                history_back.push_back(move);
                pop();

                moves.push_back(move.NaturalOut(&board));
            }

            // Restore state
            while (history_back.size())
            {
                push(encode(history_back.back()));
                history_back.pop_back();
            }

            // Walk in reverse through the generated moves.
            int mn = 1;

            while (moves.size()) {
                output += std::to_string(mn) + ". ";
                output += moves.back();
                moves.pop_back();

                if (!moves.size()) break;

                output += " " + moves.back() + " ";
                moves.pop_back();

                ++mn;
            }

            output += " " + result;
            return output;
        }
};
}

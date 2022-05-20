#pragma once

#include <iostream>

#include <vector>

#include <cstring>
#define strcpy_s(s, n, p) strncpy(s, p, n)
#include "chess/thc.h"

namespace kami {
constexpr int NFEATURES = 1 + 8 + 6 + 4 + 12;

class Env {
    private:
        thc::ChessRules board;
        float curturn;
        std::vector<thc::Move> history;

    public:
        Env() {
            board.Init();
            curturn = 1.0f;
        }

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
            float header[1 + 8 + 6 + 4];

            header[0] = 0.5f + curturn / 2.0f;

            for (int i = 0; i < 8; ++i)
                header[1 + i] = (history.size() >> (i + 1)) & 1;

            for (int i = 0; i < 6; ++i)
                header[9 + i] = (board.half_move_clock >> i) & 1;

            header[15] = board.wking_allowed() ? 1.0f : 0.0f;
            header[16] = board.wqueen_allowed() ? 1.0f : 0.0f;
            header[17] = board.bking_allowed() ? 1.0f : 0.0f;
            header[18] = board.bqueen_allowed() ? 1.0f : 0.0f;

            for (int rank = 0; rank < 8; ++rank)
            {
                for (int file = 0; file < 8; ++file)
                {
                    float* base = dst + (rank * 8 * NFEATURES) + (file * NFEATURES);

                    for (int h = 0; h < 19; ++h)
                        base[h] = header[h];

                    char pc = board.squares[(7 - rank) * 8 + file];

                    base += 19;

                    if (pc == tolower(pc))
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
        }

        void pop()
        {
            //std::cout << "popping " << history.back().TerseOut() << std::endl;
            board.PopMove(history.back());
            history.pop_back();
            curturn = -curturn;
        }

        bool terminal(float* value)
        {
            thc::TERMINAL val;
            thc::DRAWTYPE unused;

            board.Evaluate(val);

            if (val == thc::TERMINAL_WCHECKMATE)
            {
                *value = 1.0f;
                return true;
            }

            if (val == thc::TERMINAL_BCHECKMATE)
            {
                *value = -1.0f;
                return true;
            }

            if (val != thc::NOT_TERMINAL || board.IsDraw(true, unused))
            {
                *value = 0.0f;
                return true;
            }

            return false;
        }

        float turn()
        {
            return curturn;
        }

        std::vector<int> actions()
        {
            std::vector<thc::Move> moves;
            std::vector<int> out;

            board.GenLegalMoveList(moves);

            for (auto& m : moves)
                out.push_back(encode(m));

            return out;
        }

        std::string print()
        {
            return board.ToDebugStr();
        }
};
}

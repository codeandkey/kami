#pragma once

#include <atomic>
#include <iostream>

#include <vector>

#include <cstring>
#define strcpy_s(s, n, p) strncpy(s, p, n)
#include "chess/thc/thc.h"

extern "C" {
#include "chess/neocortex/types.h"
#include "chess/neocortex/position.h"
#include "chess/neocortex/attacks.h"
}

namespace kami {
constexpr int NFEATURES = 8 + 6 + 4 + 12;
constexpr int PSIZE = 73 * 64;
constexpr int WIDTH = 8;
constexpr int HEIGHT = 8;
constexpr int OBSIZE = WIDTH * HEIGHT * NFEATURES;

class NCInit {
    public: NCInit() {
        static std::atomic<bool> initialized = false;

        if (!initialized.exchange(true)) {
            ncAttacksInit();
            ncBitboardInitBetween();
            ncBitboardInitRays();
            ncZobristInit();
            std::cout << "Initialized neocortex lookup tables" << std::endl;
        }
    }
};

static NCInit nc_initializer;

class Env {
    private:
        float curturn;
        std::vector<ncMove> history;
        std::vector<int> cur_actions;
        bool actions_utd;

        ncPosition game;

    public:

        Env() {
            curturn = 1.0f;
            actions_utd = false;
            ncPositionInit(&game);
        }

        int ply() { return history.size(); }

        int encode(ncMove move) 
        {
            assert(ncMoveValid(move));

            int base;

            ncSquare src = ncMoveSrc(move);
            ncSquare dst = ncMoveDst(move);
            ncPiece srcpiece = ncBoardGetPiece(&game.board, src);
            ncPiece ptype = ncMovePtype(move);

            if (ncPositionGetCTM(&game) == NC_BLACK)
            {
                // Flip POV when black
                src = 63 - src;
                dst = 63 - dst;
            }

            int srcrank = ncSquareRank(src);
            int dstrank = ncSquareRank(dst);
            int srcfile = ncSquareFile(src);
            int dstfile = ncSquareFile(dst);

            switch (ncPieceType(srcpiece))
            {
                case NC_PAWN:
                    if (ncPieceTypeValid(ptype) && ptype != NC_QUEEN)
                    {
                        switch (ptype)
                        {
                            case NC_KNIGHT:
                                return 73 * src + 64 + (dstfile - srcfile) + 1;
                            case NC_BISHOP:
                                return 73 * src + 64 + (dstfile - srcfile) + 4;
                            case NC_ROOK:
                                return 73 * src + 64 + (dstfile - srcfile) + 7;
                        }
                    }
                case NC_QUEEN:
                case NC_ROOK:
                case NC_BISHOP:
                case NC_KING:
                {
                    int ind = ncBitboardPopcnt(ncBitboardBetween(src, dst));
                    ncBitboard dstmask = ncSquareMask(dst);

                    if (ncBitboardRay(src, NC_NORTH) & dstmask)
                        return 73 * src + ind;
                    else if (ncBitboardRay(src, NC_SOUTH) & dstmask)
                        return 73 * src + 7 + ind;
                    else if (ncBitboardRay(src, NC_EAST) & dstmask)
                        return 73 * src + 14 + ind;
                    else if (ncBitboardRay(src, NC_WEST) & dstmask)
                        return 73 * src + 21 + ind;
                    else if (ncBitboardRay(src, NC_NORTHEAST) & dstmask)
                        return 73 * src + 28 + ind;
                    else if (ncBitboardRay(src, NC_NORTHWEST) & dstmask)
                        return 73 * src + 35 + ind;
                    else if (ncBitboardRay(src, NC_SOUTHEAST) & dstmask)
                        return 73 * src + 42 + ind;
                    else
                        return 73 * src + 49 + ind;
                }
                case NC_KNIGHT:
                {
                    int ind = 0;

                    // Order W-NW, N-NW, E-NE, N-NE, W-SW, S-SW, E-SE, S-SE

                    if (srcrank > dstrank)
                        ind += 4;

                    if (srcfile < dstfile)
                        ind += 2;

                    ind += abs(srcrank - dstrank) - 1;

                    return 73 * src + 56 + ind;
                }
                default:
                    assert(0);
                    return 0;
            }
        }

        ncMove decode(int action)
        {
            assert(action >= 0 && action < PSIZE);

            ncSquare src = action / 73;
            int atype = action % 73;

            ncSquare dst;

            if (atype < 56)
            {
                // Ray move
                int dirs[] = { NC_NORTH, NC_SOUTH, NC_EAST, NC_WEST, NC_NORTHEAST, NC_NORTHWEST, NC_SOUTHEAST, NC_SOUTHWEST };
                dst = src + dirs[atype / 7] * ((atype % 7) + 1);
            }
            else if (atype < 64)
            {
                // Knight move
                int dirs[] = {
                    NC_WEST + NC_NORTHWEST,
                    NC_NORTH + NC_NORTHWEST,
                    NC_EAST + NC_NORTHEAST,
                    NC_NORTH + NC_NORTHEAST,
                    NC_WEST + NC_SOUTHWEST,
                    NC_SOUTH + NC_SOUTHWEST,
                    NC_EAST + NC_SOUTHEAST,
                    NC_SOUTH + NC_SOUTHEAST,
                };

                dst = src + dirs[atype - 56];
            }
            else
            {
                // Minor promoting move
                int dirs[] = { NC_NORTHWEST, NC_NORTH, NC_NORTHEAST };
                int ptypes[] = { NC_KNIGHT, NC_BISHOP, NC_ROOK };
                
                dst = src + dirs[(atype - 64) % 3];

                if (ncPositionGetCTM(&game) == NC_BLACK)
                {
                    src = 63 - src;
                    dst = 63 - dst;
                }

                return ncMoveMakeP(src, dst, ptypes[(atype - 64) / 3]);
            }

            if (ncPositionGetCTM(&game) == NC_BLACK)
            {
                src = 63 - src;
                dst = 63 - dst;
            }

            return ncMoveMake(src, dst);
        }

        void observe(float* dst) {
            float header[8 + 6 + 4];
            ncColor our_col = ncPositionGetCTM(&game);

            // Clear observation
            memset(dst, 0, sizeof(float) * OBSIZE);

            // Build square headers
            int ply = history.size();

            for (int i = 0; i < 8; ++i)
                header[i] = (ply >> i) & 1;

            int hmc = ncPositionHalfmoveClock(&game);
            for (int i = 0; i < 6; ++i)
                header[8 + i] = (hmc >> i) & 1;

            int our_k = NC_CASTLE_WHITE_K;
            int our_q = NC_CASTLE_WHITE_Q;
            int opp_k = NC_CASTLE_BLACK_K;
            int opp_q = NC_CASTLE_BLACK_Q;

            if (our_col == NC_BLACK)
            {
                our_k = NC_CASTLE_BLACK_K;
                our_q = NC_CASTLE_BLACK_Q;
                opp_k = NC_CASTLE_WHITE_K;
                opp_q = NC_CASTLE_WHITE_Q;
            }

            header[14] = game.ply[game.nply - 1].castle_rights & our_k;
            header[15] = game.ply[game.nply - 1].castle_rights & our_q;
            header[16] = game.ply[game.nply - 1].castle_rights & opp_k;
            header[17] = game.ply[game.nply - 1].castle_rights & opp_q;

            // Write all square headers
            for (int sq = 0; sq < 64; ++sq)
                memcpy(dst + sq * NFEATURES, header, sizeof(header));

            for (int rank = 0; rank < 8; ++rank)
            {
                for (int file = 0; file < 8; ++file)
                {
                    int sq = ncSquareAt(rank, file);
                    int povsq = (our_col == NC_BLACK) ? 63 - sq : sq;

                    float* base = dst + NFEATURES * povsq + 18;

                    // Note: board is flipped vertically due to THC but this should be OK anyway
                    ncPiece pc = ncBoardGetPiece(&game.board, sq);

                    if (!ncPieceValid(pc))
                        continue;

                    if (ncPieceColor(pc) != our_col)
                        base += 6;

                    base[ncPieceType(pc)] = 1.0f;
                }
            }
        }

        void push(int action)
        {
            ncMove mv = decode(action);
            ncPositionMakeMove(&game, mv);
            history.push_back(mv);
            curturn = -curturn;
            actions_utd = false;
        }

        void pop()
        {
            ncPositionUnmakeMove(&game);
            history.pop_back();
            curturn = -curturn;
            actions_utd = false;
        }

        std::string debug_action(int action)
        {
            char uci[6];
            ncMoveUCI(decode(action), uci);
            return uci;
        }

        bool terminal_str(float* value, std::string& out)
        {
            // 50-move rule
            if (ncPositionHalfmoveClock(&game) >= 50)
            {
                *value = 0;
                out = "Draw by 50-move rule";
                return true;
            }

            // Threefold repetition (usually faster than movegen)
            if (ncPositionRepCount(&game) > 3)
            {
                *value = 0;
                out = "Draw by threefold repetition";
                return true;
            }

            // Insufficient material
            ncBitboard kings, knights, bishops, global, white, black;

            kings = ncBoardPieceOcc(&game.board, NC_KING);
            knights = ncBoardPieceOcc(&game.board, NC_KNIGHT);
            bishops = ncBoardPieceOcc(&game.board, NC_BISHOP);
            global = ncBoardGlobalOcc(&game.board);
            white = ncBoardColorOcc(&game.board, NC_WHITE);
            black = ncBoardColorOcc(&game.board, NC_BLACK);

            if (
                kings == global // K vs K

                || (global == (kings | bishops) && ( // K/B endgame
                        ncBitboardPopcnt(bishops) == 1 // K vs KB
                        || (ncBitboardPopcnt(white) == ncBitboardPopcnt(black) && ncBitboardPopcnt(bishops) == 2) // KB vs KB
                   ))

                || (global == (kings | knights) && ( // K/N endgame
                        ncBitboardPopcnt(knights) == 1 // K vs KN
                        || (ncBitboardPopcnt(white) == ncBitboardPopcnt(black) && ncBitboardPopcnt(knights) == 2) // KN vs KN
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

            if (ncPositionIsCheck(&game))
            {
                if (ncPositionGetCTM(&game) == NC_WHITE)
                {
                    *value = -1.0f;
                    out = "White is checkmated";
                } else
                {
                    *value = 1.0f;
                    out = "Black is checkmated";
                }

                // DEBUG crazy terminals
                /*
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
                }*/

                return true;
            }

            *value = 0.0f;

            if (ncPositionGetCTM(&game) == NC_WHITE)
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
                ncMove moves[NC_MAX_PL_MOVES];
                int n = ncPositionPLMoves(&game, moves);
                ncPositionOrderMoves(&game, moves, n);

                cur_actions.clear();

                // Test if at least one legal move
                for (int i = 0; i < n; ++i)
                {
                    int legal = ncPositionMakeMove(&game, moves[i]);
                    ncPositionUnmakeMove(&game);

                    if (legal)
                        cur_actions.push_back(encode(moves[i]));
                }

                actions_utd = true;
            }

            return cur_actions;
        }

        std::string print()
        {
            char fen[100];
            ncPositionToFen(&game, fen, sizeof(fen));
            return fen;
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
                char uci[6];
                ncMoveUCI(mv, uci);
                move.TerseIn(&board, uci);
                output += " " + move.NaturalOut(&board);

                board.PushMove(move);
            }

            return output + " " + result;
        }

        float bootstrap_value(float window)
        {
            float score = (float) ncPositionEvaluate(&game) / window;
            
            score = std::min(score, 1.0f);
            score = std::max(score, -1.0f);

            return score;
        }
};
}

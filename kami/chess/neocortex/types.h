#pragma once

#include <assert.h>
#include <stdint.h>

typedef uint64_t ncBitboard;
typedef int      ncColor;
typedef uint64_t ncHashKey;
typedef int      ncMove;
typedef int      ncPiece;
typedef int      ncSquare;

#define NC_RANK_1 0xFFULL
#define NC_RANK_2 (0xFFULL << 8)
#define NC_RANK_3 (0xFFULL << 16)
#define NC_RANK_4 (0xFFULL << 24)
#define NC_RANK_5 (0xFFULL << 32)
#define NC_RANK_6 (0xFFULL << 40)
#define NC_RANK_7 (0xFFULL << 48)
#define NC_RANK_8 (0xFFULL << 56)

#define NC_FILE_A 0x0101010101010101ULL
#define NC_FILE_B (NC_FILE_A << 1)
#define NC_FILE_C (NC_FILE_A << 2)
#define NC_FILE_D (NC_FILE_A << 3)
#define NC_FILE_E (NC_FILE_A << 4)
#define NC_FILE_F (NC_FILE_A << 5)
#define NC_FILE_G (NC_FILE_A << 6)
#define NC_FILE_H (NC_FILE_A << 7)

#define NC_EAST 1
#define NC_WEST -1
#define NC_NORTH 8
#define NC_SOUTH -8
#define NC_NORTHEAST 9
#define NC_NORTHWEST 7
#define NC_SOUTHEAST -7
#define NC_SOUTHWEST -9

#define NC_WHITE 0
#define NC_BLACK 1

#define NC_NULL -1
#define NC_PAWN 0
#define NC_KNIGHT 1
#define NC_BISHOP 2
#define NC_ROOK 3
#define NC_QUEEN 4
#define NC_KING 5

static const ncBitboard NC_NEIGHBOR_FILES[8] =
{
    NC_FILE_B,
    NC_FILE_A | NC_FILE_C,
    NC_FILE_B | NC_FILE_D,
    NC_FILE_C | NC_FILE_E,
    NC_FILE_D | NC_FILE_F,
    NC_FILE_E | NC_FILE_G,
    NC_FILE_F | NC_FILE_H,
    NC_FILE_G
};

extern ncBitboard NC_BETWEEN[64][64];

void ncBitboardInitBetween();
void ncBitboardInitRays();
ncBitboard ncBitboardBetween(ncSquare src, ncSquare dst);
ncBitboard ncBitboardRay(ncSquare src, int dir);

/**
 * Locates the position of the least significant '1' bit in a bitboard.
 * Equivalent to locating the "next" square in a set.
 *
 * @param b Input bitboard. Must have at least one square set.
 * @return int Least significant square set in bitboard.
 */
static inline int ncBitboardUnmask(ncBitboard b)
{
    unsigned long pos;
    assert(b);

#ifdef _WIN32 
    _BitScanForward64(&pos, b);
#else
    pos = __builtin_ctzll(b);
#endif

    return (int) pos;
}

/**
 * Returns the population count of a bitboard.
 *
 * @param b Input bitboard.
 * @return Number of '1' bits in the input.
 */
static inline int ncBitboardPopcnt(ncBitboard b)
{
#ifdef _WIN32 
    return (int) __popcnt64(b);
#else
    return __builtin_popcountll(b);
#endif
}

/**
 * Pops a square from a bitboard.
 *
 * @param b Input bitboard. Must have at least one square set.
 * @return int Least significant square set in bitboard.
 */
static inline int ncBitboardPop(ncBitboard* b)
{
    int pos = ncBitboardUnmask(*b);
    *b ^= (1ULL << pos);
    return (int) pos;
}

static inline ncBitboard ncBitboardShift(ncBitboard b, int dir)
{
    return (dir > 0) ? b << dir : b >> -dir;
}

static inline int ncColorValid(ncColor col)
{
    return !(col / 2); // what
}

static inline int ncPieceValid(ncPiece p)
{
    return p >= 0 && p < 12;
}

static inline int ncPieceTypeValid(ncPiece p)
{
    return p >= 0 && p < 6;
}

static inline ncColor ncPieceColor(ncPiece p)
{
    assert(ncPieceValid(p));
    return p & 1;
}

static inline ncPiece ncPieceMake(ncPiece ptype, ncColor col)
{
    assert(ncPieceTypeValid(ptype));
    assert(ncColorValid(col));

    return col | (ptype << 1);
}

static inline ncPiece ncPieceFromChar(char c)
{
    switch (c)
    {
        case 'p':
            return ncPieceMake(NC_PAWN, NC_BLACK);
        case 'n':
            return ncPieceMake(NC_KNIGHT, NC_BLACK);
        case 'b':
            return ncPieceMake(NC_BISHOP, NC_BLACK);
        case 'r':
            return ncPieceMake(NC_ROOK, NC_BLACK);
        case 'q':
            return ncPieceMake(NC_QUEEN, NC_BLACK);
        case 'k':
            return ncPieceMake(NC_KING, NC_BLACK);
        case 'P':
            return ncPieceMake(NC_PAWN, NC_WHITE);
        case 'N':
            return ncPieceMake(NC_KNIGHT, NC_WHITE);
        case 'B':
            return ncPieceMake(NC_BISHOP, NC_WHITE);
        case 'R':
            return ncPieceMake(NC_ROOK, NC_WHITE);
        case 'Q':
            return ncPieceMake(NC_QUEEN, NC_WHITE);
        case 'K':
            return ncPieceMake(NC_KING, NC_WHITE);
        default:
            return NC_NULL;
    }
}

static inline char ncPieceToChar(ncPiece p)
{
    assert(ncPieceValid(p));
    return "PpNnBbRrQqKk"[p];
}

static inline char ncPieceTypeToChar(ncPiece p)
{
    assert(ncPieceValid(p));
    return "pnbrqk"[p];
}

static inline ncPiece ncPieceType(ncPiece p)
{
    assert(ncPieceValid(p));
    return p >> 1;
}

static inline ncSquare ncSquareAt(int rank, int file)
{
    assert(rank >= 0 && rank < 8);
    assert(file >= 0 && file < 8);

    return rank * 8 + file;
}

static inline int ncSquareValid(ncSquare s)
{
    return s >= 0 && s < 64;
}

static inline int ncSquareFile(ncSquare s)
{
    assert(ncSquareValid(s));
    return s % 8;
}

static inline ncBitboard ncSquareNeighborFiles(ncSquare sq)
{
    return NC_NEIGHBOR_FILES[ncSquareFile(sq)];
}

static inline int ncSquareRank(ncSquare s)
{
    assert(ncSquareValid(s));
    return s / 8;
}

static inline ncBitboard ncSquareMask(ncSquare s)
{
    assert(ncSquareValid(s));
    return 1ULL << s;
}

static inline int ncMoveValid(ncMove mv)
{
    return mv > 0 && mv < 0xffff;
}

static inline ncMove ncMoveMake(ncSquare src, ncSquare dst)
{
    assert(ncSquareValid(src));
    assert(ncSquareValid(dst));

    return src << 6 | dst | 0xF000;
}

static inline ncMove ncMoveMakeP(ncSquare src, ncSquare dst, ncPiece ptype)
{
    assert(ncPieceTypeValid(ptype));
    return src << 6 | dst | (ptype << 12); 
}

static inline ncSquare ncMoveSrc(ncMove mv)
{
    assert(ncMoveValid(mv));
    return (mv >> 6) & 0x3f;
}

static inline ncSquare ncMoveDst(ncMove mv)
{
    assert(ncMoveValid(mv));
    return mv & 0x3f;
}

static inline ncPiece ncMovePtype(ncMove mv)
{
    return (mv >> 12) & 0xF;
}

static inline void ncMoveUCI(ncMove mv, char* dst)
{
    assert(ncMoveValid(mv));

    dst[0] = ncSquareFile(ncMoveSrc(mv)) + 'a';
    dst[1] = ncSquareRank(ncMoveSrc(mv)) + '1';
    dst[2] = ncSquareFile(ncMoveDst(mv)) + 'a';
    dst[3] = ncSquareRank(ncMoveDst(mv)) + '1';

    dst[4] = dst[5] = '\0';

    if (ncPieceTypeValid(ncMovePtype(mv)))
        dst[4] = ncPieceTypeToChar(ncMovePtype(mv));
}

static inline ncMove ncMoveFromUci(char* uci)
{
    int srcfile, srcrank, dstfile, dstrank;
    int ptype = 0xF;

    srcfile = uci[0] - 'a';
    srcrank = uci[1] - '1';
    dstfile = uci[2] - 'a';
    dstrank = uci[3] - '1';

    if (uci[4])
    {
        ptype = ncPieceType(ncPieceFromChar(uci[4]));
        if (!ncPieceTypeValid(ptype)) return -1;
    }

    if (srcfile < 0 || srcfile >= 8 || dstfile < 0 || dstfile >= 8) return -1;
    if (srcrank < 0 || srcrank >= 8 || dstrank < 0 || dstrank >= 8) return -1;

    ncSquare src = ncSquareAt(srcrank, srcfile);
    ncSquare dst = ncSquareAt(dstrank, dstfile);

    return src << 6 | dst | (ptype << 12); 
}

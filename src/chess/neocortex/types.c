#include "types.h"
#include "attacks.h"

#include <string.h>

ncBitboard NC_BETWEEN[64][64];
ncBitboard NC_RAYS[64][8];
static int between_init = 0;

void ncBitboardInitBetween()
{
    memset(NC_BETWEEN, 0, sizeof NC_BETWEEN);

    for (ncSquare src = 0; src < 64; ++src)
    {
        ncBitboard dstsqs = ncAttacksQueen(src, 0ULL);

        while (dstsqs)
        {
            ncSquare dst = ncBitboardPop(&dstsqs);
			ncBitboard between = 0ULL;

            ncSquare start = src;

            while (start != dst)
            {
                int shift = 0;

                if (ncSquareRank(start) < ncSquareRank(dst))
                    shift += NC_NORTH;
                else if (ncSquareRank(start) > ncSquareRank(dst))
                    shift += NC_SOUTH;

                if (ncSquareFile(start) < ncSquareFile(dst))
                    shift += NC_EAST;
                else if (ncSquareFile(start) > ncSquareFile(dst))
                    shift += NC_WEST;

                start += shift;
                between |= ncSquareMask(start);
            }

            NC_BETWEEN[src][dst] = between & ~ncSquareMask(dst);
        }
    }

    between_init = 1;
}

void ncBitboardInitRays()
{
    memset(NC_RAYS, 0, sizeof(NC_RAYS));
    for (ncSquare src = 0; src < 64; ++src)
    {
        // North
        for (ncSquare sq = src + NC_NORTH; ncSquareValid(sq); sq += NC_NORTH)
            NC_RAYS[src][0] |= ncSquareMask(sq);

        // South 
        for (ncSquare sq = src + NC_SOUTH; ncSquareValid(sq); sq += NC_SOUTH)
            NC_RAYS[src][1] |= ncSquareMask(sq);

        // East 
        for (ncSquare sq = src + NC_EAST; ncSquareValid(sq) && ncSquareFile(sq) > ncSquareFile(src); sq += NC_EAST)
            NC_RAYS[src][2] |= ncSquareMask(sq);

        // West 
        for (ncSquare sq = src + NC_WEST; ncSquareValid(sq) && ncSquareFile(sq) < ncSquareFile(src); sq += NC_WEST)
            NC_RAYS[src][3] |= ncSquareMask(sq);

        // Northeast 
        for (ncSquare sq = src + NC_NORTHEAST; ncSquareValid(sq) && ncSquareFile(sq) > ncSquareFile(src); sq += NC_NORTHEAST)
            NC_RAYS[src][4] |= ncSquareMask(sq);

        // Northwest 
        for (ncSquare sq = src + NC_NORTHWEST; ncSquareValid(sq) && ncSquareFile(sq) < ncSquareFile(src); sq += NC_NORTHWEST)
            NC_RAYS[src][5] |= ncSquareMask(sq);

        // Southeast 
        for (ncSquare sq = src + NC_SOUTHEAST; ncSquareValid(sq) && ncSquareFile(sq) > ncSquareFile(src); sq += NC_SOUTHEAST)
            NC_RAYS[src][6] |= ncSquareMask(sq);

        // Southwest 
        for (ncSquare sq = src + NC_SOUTHWEST; ncSquareValid(sq) && ncSquareFile(sq) < ncSquareFile(src); sq += NC_SOUTHWEST)
            NC_RAYS[src][7] |= ncSquareMask(sq);
    }
}

ncBitboard ncBitboardBetween(ncSquare src, ncSquare dst)
{
    assert(ncSquareValid(src));
    assert(ncSquareValid(dst));
    assert(between_init);

    return NC_BETWEEN[src][dst];
}

ncBitboard ncBitboardRay(ncSquare src, int dir)
{
    assert(ncSquareValid(src));

    switch (dir)
    {
        case NC_NORTH:
            return NC_RAYS[src][0];
        case NC_SOUTH:
            return NC_RAYS[src][1];
        case NC_EAST:
            return NC_RAYS[src][2];
        case NC_WEST:
            return NC_RAYS[src][3];
        case NC_NORTHEAST:
            return NC_RAYS[src][4];
        case NC_NORTHWEST:
            return NC_RAYS[src][5];
        case NC_SOUTHEAST:
            return NC_RAYS[src][6];
        case NC_SOUTHWEST:
            return NC_RAYS[src][7];
        default:
            assert(0);
            return 0ULL;
    }
}

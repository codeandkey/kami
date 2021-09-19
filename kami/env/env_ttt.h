#ifndef KAMI_ENV_TTT_H
#define KAMI_ENV_TTT_H

/**
 * Tic-tac-toe environment type.
 */

#include "env.h"

#include <stdlib.h>

typedef struct {
    int squares[9]; // 0: empty, 1: X, -1: O
    int actions[9];
    int num_actions;
    int turn; // 0: X turn, 1: O turn
} tttState;

static void* tttAlloc() {
    tttState* state = malloc(sizeof(state));

    memset(state, 0, sizeof(state));

    return (void*) state;
}

static void tttPush(void* ptr, int action) {
    tttState* self = (tttState*) ptr;

    self->squares[action] = (self->turn * -2) + 1;
    self->turn = !self->turn;
    self->actions[self->num_actions++] = action;
}

static void tttPop(void* ptr) {
    tttState* self = (tttState*) ptr;

    self->squares[self->actions[self->num_actions - 1]] = 0;
    self->num_actions--;
    self->turn = !self->turn;
}

static int* tttGenerate(void* self, int* count) {
    tttState* self = (tttState*) ptr;

    int* aout = malloc(9 * sizeof(int));
    *count = 0;

    for (int i = 0; i < 9; ++i) {
        if (!self->squares[i]) {
            aout[(*count)++] = i;
        }
    }

    return aout;
}

static int* tttTerminal(void* self, float* value) {

}

static const kamiEnv kamiEnv_TicTacToe = {
    .policy_size = 9,

};

#endif

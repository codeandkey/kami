#ifndef KAMI_ENV_H
#define KAMI_ENV_H

/**
 * Environment types.
 *
 * These structures manage the environment which the agent trains in.
 * This should define entirely the "game" which is being played.
 */

typedef struct {
    int policy_size;

    void* (*alloc)();
    void (*push)(void* self, int action);
    void (*pop)(void* self);
    int (*generate)(void* self, int* actions);
    int (*terminal)(void* self, float* value);
} kamiEnv;

#endif

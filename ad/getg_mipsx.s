// +build mips mipsle

#include "textflag.h"

TEXT ·getg(SB), NOSPLIT, $0-4
    MOVW    g, R8
    MOVW    R8, ret+0(FP)
    RET

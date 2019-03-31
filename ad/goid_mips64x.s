// +build mips64 mips64le

#include "textflag.h"

TEXT ·getg(SB), NOSPLIT, $0-8
    MOVD    g, R8
    MOVD    R8, ret+0(FP)
    RET

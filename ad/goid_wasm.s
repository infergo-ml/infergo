// Copyright 2016 Huan Du. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

// getg is not yet supported for WebAssembly.
// I need to figure out how to implement it.
TEXT ·getg(SB), NOSPLIT, $0-4
    RET

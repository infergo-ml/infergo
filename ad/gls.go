package ad

// Multi-threaded tape store, suitable for running
// multiple goroutines with inference in parallel.

import (
	"bytes"
	"fmt"
	"runtime"
	"sync"
)

type mtStore struct {
}

var traceBuf = sync.Pool {
	New: func() interface{} {
		buf := make([]byte, 64)
		return &buf
	},
}

// goID returns the current goroutine id as a string.
func goID() string {
	bp := traceBuf.Get().(*[]byte)
	defer traceBuf.Put(bp)
	b := *bp
	b = b[len("goroutine "):runtime.Stack(b, false)]
	// Parse the 4707 out of "goroutine 4707 ["
	i := bytes.IndexByte(b, ' ')
	if i < 0 {
		panic(fmt.Sprintf("No space found in %q", b))
	}
	return string(b[:i])
}

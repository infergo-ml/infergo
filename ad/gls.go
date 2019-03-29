package ad

// Multi-threaded tape store, suitable for running
// multiple goroutines with inference in parallel.

import (
	"bytes"
	"fmt"
	"runtime"
	"sync"
)

type mtStore map[string]*adTape

// MTSafeOn makes differentiation thread safe at
// the expense of a loss in performance. There is
// no corresponding MTSafeOff, as once things are
// safe they cannot safely become unsafe again.
func MTSafeOn() {
	tapes = &mtStore{}
	mtSafe = true
}

func (tapes *mtStore) get() *adTape {
	id := goID()
	tape, ok := (*tapes)[id]
	if !ok {
		tape := newTape()
		(*tapes)[id] = tape
	}
	return tape
}

func(tapes *mtStore) drop() {
	id := goID()
	delete(*tapes, id)
}

func(tapes *mtStore) clear() {
	for key := range *tapes {
		delete(*tapes, key)
	}
}

// We need to obtain the goroutine id. The trick below works
// with go 1.11, and will hopefully continue to work.
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
	// b == "goroutine 1234 ..."
	b = b[len("goroutine "):runtime.Stack(b, false)]
	i := bytes.IndexByte(b, ' ')
	if i < 0 {
		panic(fmt.Sprintf("Cannot extract goID from %q", *bp))
	}
	return string(b[:i])
}

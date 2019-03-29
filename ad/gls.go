package ad

// Multi-threaded tape store, suitable for running
// multiple goroutines with inference in parallel.

import (
	"bytes"
	"fmt"
	"runtime"
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
		tape = newTape()
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

// goID returns the current goroutine id as a string.
func goID() string {
	b := make([]byte, 64)
	// b == "goroutine 1234 ..."
	id := b[len("goroutine "):runtime.Stack(b, false)]
	i := bytes.IndexByte(id, ' ')
	if i < 0 {
		panic(fmt.Sprintf("Cannot extract goID from %q", b))
	}
	id = id[:i]
	return string(id)
}

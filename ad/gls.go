package ad

// Multi-threaded tape store, suitable for running
// multiple goroutines with inference in parallel.

import (
	"github.com/modern-go/gls"
)

type mtStore map[int64]*adTape

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

// goID returns the current goroutine id.
func goID() int64 {
	return gls.GoID()
}

package ad

// Multi-threaded tape store, suitable for running multiple
// goroutines with inference in parallel.

import (
	"sync"
)

// sync.Map is slightly slower in a single thread, but much
// better when multiple goroutines are running concurrently.
type mtStore struct {
	sync.Map
}

func newStore() *mtStore {
	return &mtStore{}
}

// MTSafeOn makes differentiation thread safe at the expense of
// a loss in performance. There is no corresponding MTSafeOff,
// as once things are safe they cannot safely become unsafe
// again.
func MTSafeOn() {
	tapes = newStore()
	mtSafe = true
}

func (tapes *mtStore) get() *adTape {
	id := goid()
	tape, ok := tapes.Load(id)
	if !ok {
		tape = newTape()
		tapes.Store(id, tape)
	}
	return tape.(*adTape)
}

func(tapes *mtStore) drop() {
	id := goid()
	tapes.Delete(id)
}

func(_ *mtStore) clear() {
	tapes = newStore()
}

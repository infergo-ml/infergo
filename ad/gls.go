package ad

// Multi-threaded tape store, suitable for running
// multiple goroutines with inference in parallel.

import (
	"github.com/dtolpin/gls"
	"sync"
)

type mtStore struct {
	store map[int64]*adTape
	mutex sync.Mutex
}

// MTSafeOn makes differentiation thread safe at
// the expense of a loss in performance. There is
// no corresponding MTSafeOff, as once things are
// safe they cannot safely become unsafe again.
func MTSafeOn() {
	tapes = &mtStore{
		store: map[int64]*adTape{},
		mutex: sync.Mutex{},
	}
	mtSafe = true
}

func (tapes *mtStore) get() *adTape {
	id := goID()
	tapes.mutex.Lock()
	tape, ok := tapes.store[id]
	tapes.mutex.Unlock()
	if !ok {
		tape = newTape()
		tapes.mutex.Lock()
		tapes.store[id] = tape
		tapes.mutex.Unlock()
	}
	return tape
}

func(tapes *mtStore) drop() {
	id := goID()
	tapes.mutex.Lock()
	delete(tapes.store, id)
	tapes.mutex.Unlock()
}

func(tapes *mtStore) clear() {
	for key := range tapes.store {
		tapes.mutex.Lock()
		delete(tapes.store, key)
		tapes.mutex.Unlock()
	}
}

// goID returns the current goroutine id.
func goID() int64 {
	return gls.GoID()
}

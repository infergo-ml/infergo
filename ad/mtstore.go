package ad

// Multi-threaded tape store, suitable for running multiple
// goroutines with inference in parallel.

import (
	"runtime"
	"sync"
)

// sync.Map is slightly slower than map and mutex in a single
// goroutine, but much better when multiple goroutines are
// running concurrently.
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
//
// MTSafeOn enables multithreading support on some versions and
// architectures only. The caller should check the return value
// (true if succeeded) if the code depends on the tape being
// thread-safe.
func MTSafeOn() bool {
	switch runtime.GOARCH {
	case "386", "amd64p32", "amd64", "arm", "arm64", "wasm":
		tapes = newStore()
		mtSafe = true
	default:
		// not supported
	}

	return mtSafe
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

func (tapes *mtStore) drop() {
	id := goid()
	tapes.Delete(id)
}

func (_ *mtStore) clear() {
	tapes = newStore()
}

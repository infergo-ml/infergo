package ad

// Multi-threaded tape store, suitable for running multiple
// goroutines with inference in parallel.

import (
	"log"
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

var mtSafe = false

// IsMTSafe returns true if multithreading support is turned on,
// and multiple differentiations may run concurrently.
func IsMTSafe() bool {
	return mtSafe
}

var warnedNoMT = false

// MTSafeOn makes differentiation thread safe at the expense of
// a loss in performance. There is no corresponding MTSafeOff,
// as once things are safe they cannot safely become unsafe
// again.
//
// MTSafeOn enables multithreading support on some versions and
// architectures only. The caller should check the return value
// (true if succeeded) or call IsMTSafe if the code depends on
// the tape being thread-safe.
func MTSafeOn() bool {
	switch runtime.GOARCH {
	case "386", "amd64p32", "amd64", "arm", "arm64", "wasm":
		tapes = newStore()
		mtSafe = true
	case "mips", "mipsle", "mips64", "mips64le",
		"ppc64", "ppc64le", "s390x":
		if !warnedNoMT {
			log.Printf("WARNING: multithreading was not tested "+
				"on %s.", runtime.GOARCH)
			warnedNoMT = true
			tapes = newStore()
			mtSafe = true
		}
	default:
		if !warnedNoMT {
			log.Printf("WARNING: multithreading is not supported "+
				"on %s.", runtime.GOARCH)
			warnedNoMT = true
		}
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

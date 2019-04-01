package ad

// Multi-threaded tape store, suitable for running multiple
// goroutines with inference in parallel.

import (
	"log"
	"regexp"
	"runtime"
	"strconv"
	"sync"
	"unsafe"
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
	if atleast(runtime.Version(), 1, 9, 0) {
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
	} else {
		if !warnedNoMT {
			log.Printf("WARNING: multithreading is not supported "+
				"for Go version %s.", runtime.Version())
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

var goidOffset uintptr

func init() {
	switch {
	case atleast(runtime.Version(), 1, 9, 0):
		goidOffset = 152
	default:
		// unsupported version
	}
}

// Implementation of goid

// Go assembly provides the 'g' register holding the pointer
// to current goroutine's runtime.g. Function goid calls getg
// (implemented for each supported architecture) and
// adds the offset of the goid field. The offset depends
// on the Go version.

// goid returns the goroutine id of current goroutine
func goid() int64 {
	g := getg()
	p_goid := (*int64)(unsafe.Pointer(g + goidOffset))
	return *p_goid
}

// getg returns the g pointer and is implemented in Go assembly.
func getg() uintptr

// Multithreading is set up based on Go version and
// architecture. Function atleast compares a go version to
// the base version.

// goMajorMinorPatch is a regular expression for the semantic
// versioning of Go releases
var goMajorMinorPatch = regexp.MustCompile(
	`go([0-9]+)(?:\.([0-9]+)(?:\.([0-9]+))?)?.*`)

// atleast returns true if version is at least
// major.minor.patch, false otherwise.
func atleast(goVersion string, major, minor, patch int) bool {
	matches := goMajorMinorPatch.FindStringSubmatch(goVersion)
	if matches == nil {
		// Not a semantic version, an unreleased build is always
		// the latest one.
		return true
	}

	base := []int{major, minor, patch}
	for i, s := range matches[1:] {
		version, _ := strconv.Atoi(s)
		if base[i] > version {
			return false
		}
	}
	return true
}

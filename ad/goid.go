package ad

import (
	"runtime"
	"unsafe"
)

var goidOffset uintptr

func init() {
	version := runtime.Version()
	switch {
	case version >= "1.10": // at least since 1.10
		goidOffset = 152
	}
}

// goid returns the goroutine id of current goroutine
func goid() int64 {
	g := getg()
	p_goid := (*int64)(unsafe.Pointer(g + goidOffset))
	return *p_goid
}

func getg() uintptr
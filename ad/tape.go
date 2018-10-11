package ad

// Implementation of the AD tape

type Tape struct {
	records []record             // built during forward pass
	bars    map[*float64]float64 // filled during backward pass
	stack   []float64            // used for passing arguments
	rc, sc  int                  // record and stack counters
}

const maxarg int = 2

type record struct {
	typ, opcode int // record type and opcode
	value       float64
	parameters  [maxarg]float64
	args        [maxarg]*float64
}

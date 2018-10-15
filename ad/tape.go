package ad

import (
	"fmt"
)

// Implementation of the AD tape

// There is one global tape.
var t tape

// A tape is a list of records and the memory
type tape struct {
	records    []record             // recorded instructions
	lvalues    []*float64           // addresses of locations
	rvalues    []float64            // stored intermediate values
	elementals []uintptr            // elemental function indices
	bars       map[*float64]float64 // bars
	cstack     []counters           // counter stack (see below)
}

func init() {
	t = tape{
		records:    make([]record, 0),
		lvalues:    make([]*float64, 0),
		rvalues:    make([]float64, 0),
		elementals: make([]uintptr, 0),
		bars:       make(map[*float64]float64),
		cstack:     make([]counters, 0),
	}
}

// A record specifies the record type and indexes the tape memory
// to specify record argument. At the cost of one redirection,
// the number of memory allocations is logarithmic in the number
// of instructions, and a record has a fixed size.
type record struct {
	typ, op int //  record type and opcode or index of elemental
	lv, rv  int // indices of the first pointer and value
}

// The counters structure holds counters for the tape
// components. Counters are pushed onto stack for repeated
// calls to automatic differentiation (e.g. for nested
// inference).
type counters struct {
	i, // independents
	r, // records
	lv, // locations
	rv, // intermediate values
	e int // elementals
}

// Record types
const (
	typAssignment = iota
	typArithmetic
	typElemental
)

// Arithmetic operations
const (
	opNeg = iota
	opAdd
	opSub
	opMul
	opDiv
)

// Forward pass

// TODO

// Backward pass

// Function Graient performs the backward pass on
// the tape and returns the gradient. It should be
// called immediately after the call to an automatically
// differentiated function, and can be called only once
// per call to an automatically differentiated function.
func Gradient() []float64 {
	backward(&t)
	gradient := partials(&t)
	pop(&t)

	return gradient
}

// Function backward runs the backward pass
// on the tape.
func backward(t *tape) {
	bottom := t.cstack[len(t.cstack)-1].r
	for ir := len(t.records); ir != bottom; {
		ir--
		rec := t.records[ir]
		switch rec.typ {
		case typAssignment:
			// restore previous value
			*t.lvalues[rec.lv] = t.rvalues[rec.rv]
			// update the bars: the bar of the left-hand side 
			// is zero (because it is overwritten) except if 
			// the right-hand side is the same location.
			bar := t.bars[t.lvalues[rec.lv]]
			t.bars[t.lvalues[rec.lv]] = 0.
			t.bars[t.lvalues[rec.lv + 1]] = bar
		case typArithmetic:
			switch rec.op {
			case opNeg:
				bar = t.bars[t.lvalues[rec.lv]]
				t.bars[t.lvalues[rec.lv + 1]] = -bar
			case opAdd:
				// TODO
			case opSub:
				// TODO
			case opMul:
				// TODO
			case opDiv:
				// TODO
			default:
				panic(fmt.Sprintf("bad opcode %v", rec.op))
			}
		case typElemental:
			// TODO
		default:
			panic(fmt.Sprintf("bad type %v", rec.typ))
		}
	}
}

// Function partials collects the partial derivatives;
// first c.i locations are parameters.
func partials(t *tape) []float64 {
	c := &t.cstack[len(t.cstack)-1]
	partials := make([]float64, c.i)
	for i := 0; i != c.i; i++ {
		partials[i] = t.bars[t.lvalues[c.lv + i]]
	}
	return partials
}

// Function pop deallocates current tape fragment
// from the tape.
func pop(t *tape) {
	c := &t.cstack[len(t.cstack)-1]
	for i := c.lv; i != len(t.lvalues); i++ {
		delete(t.bars, t.lvalues[i])
	}
	t.records = t.records[:c.r]
	t.lvalues = t.lvalues[:c.lv]
	t.rvalues = t.rvalues[:c.rv]
	t.elementals = t.elementals[:c.e]
	t.cstack = t.cstack[:len(t.cstack)-1]
}

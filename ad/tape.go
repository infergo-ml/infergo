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
	places     []*float64           // addresses of places
	values     []float64            // stored intermediate values
	nargs      []int                // numbers of arguments of elementals
	elementals []elemental          // gradients of elementals
	bars       map[*float64]float64 // bars
	cstack     []counters           // counter stack (see below)
}

func init() {
	t = tape{
		records:    make([]record, 0),
		places:     make([]*float64, 0),
		values:     make([]float64, 0),
		nargs:      make([]int, 0),
		elementals: make([]elemental, 0),
		bars:       make(map[*float64]float64),
		cstack:     make([]counters, 0),
	}
}

// A record specifies the record type and indexes the tape memory
// to specify record argument. At the cost of one redirection,
// the number of memory alplaces is logarithmic in the number
// of instructions, and a record has a fixed size.
type record struct {
	typ, op int //  record type and opcode or index of gradient
	p, v    int // indices of the first pointer and value
}

// The structure elemental stores information required
// to compute the gradient.
type elemental struct {
	n int      // number of arguments
	g gradient // gradient
}

// The counters structure holds counters for the tape
// components. Counters are pushed onto stack for repeated
// calls to automatic differentiation (e.e. for nested
// inference).
type counters struct {
	n, // independents
	r, // records
	p, // places
	v, // intermediate values
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
	partials := partials(&t)
	pop(&t)

	return partials
}

// Function backward runs the backward pass
// on the tape.
func backward(t *tape) {
	bottom := t.cstack[len(t.cstack)-1].r
	for ir := len(t.records); ir != bottom; {
		ir--
		rec := &t.records[ir]
		bar := t.bars[t.places[rec.p]]
		// Only assignment may have the same location
		// on both the right-hand and the left-hand.
		switch rec.typ {
		case typAssignment: // v = u; dv/du = 1
			// Restore previous value
			*t.places[rec.p] = t.values[rec.v]
			// Update the bars: the bar of the left-hand side
			// is zero (because it is overwritten) except if
			// the right-hand side is the same location.
			t.bars[t.places[rec.p]] = 0.
			t.bars[t.places[rec.p+1]] += bar
		case typArithmetic:
			switch rec.op {
			case opNeg: // v = -u; dv/du = -1
				t.bars[t.places[rec.p+1]] -= bar
			case opAdd: // v = u + w; dv/du = 1; dv/dw = 1
				t.bars[t.places[rec.p+1]] += bar
				t.bars[t.places[rec.p+2]] += bar
			case opSub: // v = u - w; dv/du = 1; dv/dw = -1
				t.bars[t.places[rec.p+1]] += bar
				t.bars[t.places[rec.p+2]] -= bar
			case opMul: // v = u*w; dv/du = w; dv/dw = u
				baru := bar * *t.places[rec.p+2]
				barw := bar * *t.places[rec.p+1]
				t.bars[t.places[rec.p+1]] += baru
				t.bars[t.places[rec.p+2]] += barw
			case opDiv: // v = u/w; dv/du = 1/w; dv/dw = - dv/du*u
				baru := bar / *t.places[rec.p+2]
				barw := -baru * *t.places[rec.p+1]
				t.bars[t.places[rec.p+1]] += baru
				t.bars[t.places[rec.p+2]] -= barw
			default:
				panic(fmt.Sprintf("bad opcode %v", rec.op))
			}
		case typElemental: // v = f(u, w, ...)
			e := &t.elementals[rec.op]
			dv := e.g(*t.places[rec.p],
				// Parameters must be copied to t.values during
				// the forward pass.
				t.values[rec.v:rec.v+e.n]...)
			for i := 0; i != e.n; i++ {
				t.bars[t.places[rec.p+1+i]] += bar * dv[i]
			}
		default:
			panic(fmt.Sprintf("bad type %v", rec.typ))
		}
	}
}

// Function partials collects the partial derivatives;
// first c.n places are parameters.
func partials(t *tape) []float64 {
	c := &t.cstack[len(t.cstack)-1]
	partials := make([]float64, c.n)
	for i := 0; i != c.n; i++ {
		partials[i] = t.bars[t.places[c.p+i]]
	}
	return partials
}

// Function pop deallocates current tape fragment
// from the tape.
func pop(t *tape) {
	c := &t.cstack[len(t.cstack)-1]
	for i := c.p; i != len(t.places); i++ {
		delete(t.bars, t.places[i])
	}
	t.records = t.records[:c.r]
	t.places = t.places[:c.p]
	t.values = t.values[:c.v]
	t.elementals = t.elementals[:c.e]
	t.cstack = t.cstack[:len(t.cstack)-1]
}

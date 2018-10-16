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
	values     []float64            // stored values
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
	v, // values
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
	OpNeg = iota
	OpAdd
	OpSub
	OpMul
	OpDiv
)

// Forward pass

// Enter set ups the tape for the forward pass. n is the number
// of parameters of the function to to differentiate.
func Enter(n int) {
	c := counters{
		n: n,
		r: len(t.records),
		p: len(t.places),
		v: len(t.values),
		e: len(t.elementals),
	}
	t.cstack = append(t.cstack, c)
}

// Constant adds constant v to the memory and returns
// the location of the constant.
func Constant(c float64) *float64 {
	t.values = append(t.values, c)
	return &t.values[len(t.values)-1]
}

// Variable adds variable v to the memory and returns v.
func Variable(v *float64) *float64 {
	t.places = append(t.places, v)
	return v
}

// Intermediate allocates an intermediate variable
// and returns the location of the variable.
func Intermediate() *float64 {
	return Variable(Constant(0))
}

// Assigment encodes an assignment.
func Assignment(v *float64, u *float64) *float64 {
	r := record{
		typ: typAssignment,
		p:   len(t.places),
		v:   len(t.values),
	}
	t.places = append(t.places, v, u)
	t.values = append(t.values, *v)
	t.records = append(t.records, r)
	return v
}

// Arithmetic encodes an arithmetic operation and returns
// the location of the result.
func Arithmetic(op int, v *float64, u ...*float64) *float64 {
	r := record{
		typ: typArithmetic,
		op:  op,
		p:   len(t.places),
	}
	t.places = append(t.places, v)
	t.places = append(t.places, u...)
	t.records = append(t.records, r)
	return v
}

// Elemental encodes a call to the elemental f. If f is
// not registered as an elemental, the call is ignored.
// To call gradient without allocation on backward pass,
// argument values are copied to the tape memory.
// Elemental returns the location of the result.
func Elemental(f interface{}, v *float64, u ...*float64) *float64 {
	g, ok := ElementalGradient(f)
	if ok {
		// This is indeed an elemental, we have it's gradient.
		r := record{
			typ: typElemental,
			op:  len(t.elementals),
			p:   len(t.places),
			v:   len(t.values),
		}
		e := elemental{
			n: len(u),
			g: g,
		}
		t.places = append(t.places, v)
		t.places = append(t.places, u...)
		for _, w := range u {
			t.values = append(t.values, *w)
		}
		t.elementals = append(t.elementals, e)
		t.records = append(t.records, r)
	}
	return v
}

// Backward pass

// Function Graient performs the backward pass on
// the tape and returns the gradient. It should be
// called immediately after the call to an automatically
// differentiated function, and can be called only once
// per call to an automatically differentiated function.
func Gradient() []float64 {
	backward()
	partials := partials()
	pop()
	return partials
}

// Function backward runs the backward pass
// on the tape.
func backward() {
	bottom := t.cstack[len(t.cstack)-1].r
	for ir := len(t.records); ir != bottom; {
		ir--
		r := &t.records[ir]
		bar := t.bars[t.places[r.p]]
		// Only assignment may have the same location
		// on both the right-hand and the left-hand.
		switch r.typ {
		case typAssignment: // v = u; dv/du = 1
			// Restore previous value
			*t.places[r.p] = t.values[r.v]
			// Update the bars: the bar of the left-hand side
			// is zero (because it is overwritten) except if
			// the right-hand side is the same location.
			t.bars[t.places[r.p]] = 0.
			t.bars[t.places[r.p+1]] += bar
		case typArithmetic:
			switch r.op {
			case OpNeg: // v = -u; dv/du = -1
				t.bars[t.places[r.p+1]] -= bar
			case OpAdd: // v = u + w; dv/du = 1; dv/dw = 1
				t.bars[t.places[r.p+1]] += bar
				t.bars[t.places[r.p+2]] += bar
			case OpSub: // v = u - w; dv/du = 1; dv/dw = -1
				t.bars[t.places[r.p+1]] += bar
				t.bars[t.places[r.p+2]] -= bar
			case OpMul: // v = u*w; dv/du = w; dv/dw = u
				baru := bar * *t.places[r.p+2]
				barw := bar * *t.places[r.p+1]
				t.bars[t.places[r.p+1]] += baru
				t.bars[t.places[r.p+2]] += barw
			case OpDiv: // v = u/w; dv/du = 1/w; dv/dw = - dv/du*u
				baru := bar / *t.places[r.p+2]
				barw := -baru * *t.places[r.p+1]
				t.bars[t.places[r.p+1]] += baru
				t.bars[t.places[r.p+2]] -= barw
			default:
				panic(fmt.Sprintf("bad opcode %v", r.op))
			}
		case typElemental: // v = f(u, w, ...)
			e := &t.elementals[r.op]
			dv := e.g(*t.places[r.p],
				// Parameters must be copied to t.values during
				// the forward pass.
				t.values[r.v:r.v+e.n]...)
			for i := 0; i != e.n; i++ {
				t.bars[t.places[r.p+1+i]] += bar * dv[i]
			}
		default:
			panic(fmt.Sprintf("bad type %v", r.typ))
		}
	}
}

// Function partials collects the partial derivatives;
// first c.n places are parameters.
func partials() []float64 {
	c := &t.cstack[len(t.cstack)-1]
	partials := make([]float64, c.n)
	for i := 0; i != c.n; i++ {
		partials[i] = t.bars[t.places[c.p+i]]
	}
	return partials
}

// Function pop deallocates current tape fragment
// from the tape.
func pop() {
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

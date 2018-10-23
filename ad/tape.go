package ad

// Implementation of the AD tape

import (
	"fmt"
	"reflect"
)

// There is one global tape.
var t tape

// A tape is a list of records and the memory
type tape struct {
	records    []record             // recorded instructions
	places     []*float64           // variable places
	values     []float64            // stored values
	elementals []elemental          // gradients of elementals
	adjoints   map[*float64]float64 // adjoints
	cstack     []counters           // counter stack (see below)
}

func init() {
	t = tape{
		records:    make([]record, 0),
		places:     make([]*float64, 0),
		values:     make([]float64, 0),
		elementals: make([]elemental, 0),
		adjoints:   make(map[*float64]float64),
		cstack:     make([]counters, 0),
	}
	// The returned value is in the first place;
	// see Call and Return below.
	Place(Value(0.))
}

// A record specifies the record type and indexes the tape
// memory to specify the record arguments. At the cost of one
// redirection, the number of memory allocations is logarithmic
// in the number of instructions, and a record has a fixed size.
type record struct {
	typ, op int //  record type and opcode or index of gradient
	p, v    int // indices of the first place and value
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

// Setup set ups the tape for the forward pass.
func Setup(x []float64) {
	push(len(x))
	register(x)
}

// push pushes a counter frame to the counter stack.
// n is the number of function parameters.
func push(n int) {
	c := counters{
		n: n,
		r: len(t.records),
		p: len(t.places),
		v: len(t.values),
		e: len(t.elementals),
	}
	t.cstack = append(t.cstack, c)
}

// register stores locations of function parameters
// at the beginning of the current frame's places.
// The places are then used to collect the partial
// derivatives of the gradient.
func register(x []float64) {
	for i := 0; i != len(x); i++ {
		Place(&x[i])
	}
}

// Constant adds value v to the memory and returns
// the location of the value.
func Value(v float64) *float64 {
	t.values = append(t.values, v)
	return &t.values[len(t.values)-1]
}

// Variable adds place p to the memory and returns p.
func Place(p *float64) *float64 {
	t.places = append(t.places, p)
	return p
}

// Assigment encodes an assignment.
func Assignment(p *float64, px *float64) {
	// register
	r := record{
		typ: typAssignment,
		p:   len(t.places),
		v:   len(t.values),
	}
	t.places = append(t.places, p, px)
	t.values = append(t.values, *p)
	t.records = append(t.records, r)

	// run
	*p = *px
}

// Arithmetic encodes an arithmetic operation and returns
// the location of the result.
func Arithmetic(op int, px ...*float64) *float64 {
	p := Place(Value(0.))

	// register
	r := record{
		typ: typArithmetic,
		op:  op,
		p:   len(t.places),
	}
	t.places = append(t.places, p)
	t.places = append(t.places, px...)
	t.records = append(t.records, r)

	// run
	switch op {
	case OpNeg:
		*p = -*px[0]
	case OpAdd:
		*p = *px[0] + *px[1]
	case OpSub:
		*p = *px[0] - *px[1]
	case OpMul:
		*p = *px[0] * *px[1]
	case OpDiv:
		*p = *px[0] / *px[1]
	default:
		panic(fmt.Sprintf("bad opcode %v", r.op))
	}

	return p
}

// Elemental encodes a call to the elemental f.
// To call gradient without allocation on backward pass,
// argument values are copied to the tape memory.
// Elemental returns the location of the result.
func Elemental(f interface{}, px ...*float64) *float64 {
	p := Place(Value(0.))

	g, ok := ElementalGradient(f)
	if !ok {
		// actually an elemental
		panic("not an elemental")
	}

	// register
	r := record{
		typ: typElemental,
		op:  len(t.elementals),
		p:   len(t.places),
		v:   len(t.values),
	}
	e := elemental{
		n: len(px),
		g: g,
	}
	t.places = append(t.places, p)
	t.places = append(t.places, px...)
	for _, py := range px {
		t.values = append(t.values, *py)
	}
	t.elementals = append(t.elementals, e)
	t.records = append(t.records, r)

	// run
	// Any elemental function can be called, but one-
	// and two-argument elementals are called efficiently,
	// without allocation; other types are called through
	// reflection.
	switch f := f.(type) {
	case func(float64) float64:
		*p = f(*px[0])
	case func(float64, float64) float64:
		*p = f(*px[0], *px[1])
	default:
		args := make([]reflect.Value, 0)
		for _, py := range px {
			args = append(args, reflect.ValueOf(*py))
		}
		*p = reflect.ValueOf(f).Call(args)[0].Float()
	}

	return p
}

// Calling differentiated functions

// Call wraps a call to a differentiated subfunction.
func Call(f func(), px ...*float64) *float64 {
	// Register function parameters. The function
	// will assign from the actual parameters to
	// the formal parameters on entry.
	for _, py := range px {
		Place(py)
	}
	f()
	// The returned value is in the first place.
	return t.places[0]
}

// Enter copies the actual parameters to the formal parameters.
func Enter(px ...*float64) {
	i0 := len(t.places) - len(px)
	for i, py := range px {
		Assignment(py, t.places[i0+i])
	}
}

// Return returns the result of the differentiated function.
func Return(px *float64) float64 {
	// The returned value goes into the first place.
	Assignment(t.places[0], px)
	return *px
}

// Backward pass

// Gradient performs the backward pass on the tape and returns
// the gradient. It should be called immediately after the call
// to an automatically differentiated function, and can be
// called only once per call to an automatically differentiated
// function.
func Gradient() []float64 {
	backward()
	partials := partials()
	Pop()
	return partials
}

// Pop deallocates current tape fragment from the tape.
// Gradient calls Pop; when the gradient is not needed, Pop can
// be called directly to skip gradient computation.
func Pop() {
	c := &t.cstack[len(t.cstack)-1]
	for i := c.p; i != len(t.places); i++ {
		delete(t.adjoints, t.places[i])
	}
	t.records = t.records[:c.r]
	t.places = t.places[:c.p]
	t.values = t.values[:c.v]
	t.elementals = t.elementals[:c.e]
	t.cstack = t.cstack[:len(t.cstack)-1]
}

// backward runs the backward pass on the tape.
func backward() {
	r := t.records[len(t.records)-1]
	// Set the adjoint of the result to 1.
	t.adjoints[t.places[r.p]] = 1.        // set result's adjoint to 1
	bottom := t.cstack[len(t.cstack)-1].r // bottom is the first record
	for ir := len(t.records); ir != bottom; {
		ir--
		r := &t.records[ir]
		a := t.adjoints[t.places[r.p]]
		// Only assignment may have the same location
		// on both the right-hand and the left-hand.
		switch r.typ {
		case typAssignment: // x; d/dx = 1
			// Restore previous value
			*t.places[r.p] = t.values[r.v]
			// Update the adjoints: the adjoint of the left-hand side
			// is zero (because it is overwritten) except if
			// the right-hand side is the same location.
			t.adjoints[t.places[r.p]] = 0.
			t.adjoints[t.places[r.p+1]] += a
		case typArithmetic:
			switch r.op {
			case OpNeg: // -x; d/dx = -1
				t.adjoints[t.places[r.p+1]] -= a
			case OpAdd: // x + y; d/dx = 1; d/dy = 1
				t.adjoints[t.places[r.p+1]] += a
				t.adjoints[t.places[r.p+2]] += a
			case OpSub: // x - y; d/dx = 1; d/dy = -1
				t.adjoints[t.places[r.p+1]] += a
				t.adjoints[t.places[r.p+2]] -= a
			case OpMul: // x * y; d/dx = y; d/dy = x
				ax := a * *t.places[r.p+2]
				ay := a * *t.places[r.p+1]
				t.adjoints[t.places[r.p+1]] += ax
				t.adjoints[t.places[r.p+2]] += ay
			case OpDiv: // x / y; d/dx = 1 / y; d/dy = - d/dx * p
				ax := a / *t.places[r.p+2]
				ay := -ax * *t.places[r.p]
				t.adjoints[t.places[r.p+1]] += ax
				t.adjoints[t.places[r.p+2]] += ay
			default:
				panic(fmt.Sprintf("bad opcode %v", r.op))
			}
		case typElemental: // f(x, y, ...)
			e := &t.elementals[r.op]
			d := e.g(*t.places[r.p],
				// Parameters must be copied to t.values during
				// the forward pass.
				t.values[r.v:r.v+e.n]...)
			for i := 0; i != e.n; i++ {
				t.adjoints[t.places[r.p+1+i]] += a * d[i]
			}
		default:
			panic(fmt.Sprintf("bad type %v", r.typ))
		}
	}
}

// partials collects the partial derivatives; first c.n places
// are parameters.
func partials() []float64 {
	c := &t.cstack[len(t.cstack)-1]
	partials := make([]float64, c.n)
	for i := 0; i != c.n; i++ {
		partials[i] = t.adjoints[t.places[c.p+i]]
	}
	return partials
}

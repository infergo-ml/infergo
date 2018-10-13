package ad

// Implementation of the AD tape

// There is one global tape.
var t tape

func init() {
    t = tape{
        records: make([]record, 0),
        lvalues: make([]*float64, 0),
        rvalues: make([]float64, 0),
        elementals: make([]uintptr, 0),
        bars: make(map[*float64]float64),
        cstack: make([]counters, 0),
    }
}

// Function Jacobian performs the backward pass on
// the tape and returns the Jacobian.
func Jacobian() []float64 {
    c := t.cstack[len(t.cstack) - 1]

    backward(&t, &c)

    // build the Jacobian:
    // first c.i locations are parameters
    jacobian := make([]float64, c.i)
    for i := 0; i != c.i; i++ {
        jacobian[i] = t.bars[t.lvalues[c.lv + i]]
    }

    pop(&t, &c)

    return jacobian
}

// Function backward runs the backward pass 
// on the tape.
func backward(t *tape, c *counters) {
    // TODO
}

// Function pop deallocates current tape fragment 
// from the tape.
func pop(t *tape, c *counters) {
    for i := c.lv; i != len(t.lvalues); i++ {
        delete(t.bars, t.lvalues[i])
    }
    t.records = t.records[:c.r]
    t.lvalues = t.lvalues[:c.lv]
    t.rvalues = t.rvalues[:c.rv]
    t.elementals = t.elementals[:c.e]
    t.cstack = t.cstack[:len(t.cstack) - 1]
}


// A tape is a list of records and the memory
type tape struct {
	records          []record   // recorded instructions
	lvalues          []*float64 // addresses of locations
	rvalues          []float64  // stored intermediate values
	elementals       []uintptr  // elemental function indices
    bars             map[*float64]float64 // bars
    cstack           []counters // counter stack
}

// A record specifies the record type and indexes the tape memory
// to specify record argument. At the cost of one redirection,
// the number of memory allocations is logarithmic in the number
// of instructions, and a record has a fixed size.
type record struct {
	typ, op int //  record type and opcode or index of elemental
	il, ir  int // indices of the first pointer and value
}

type counters struct {
    i,        // independents
    r,        // records
    lv,       // locations
    rv,       // intermediate values
    e int     // elementals
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

// Backward pass

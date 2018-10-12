package ad

// Implementation of the AD tape

// A Tape consists of the list of records, as well as memory
// for pointers and stored intermediate values.
type Tape struct {
	records    []record   // recorded instructions
	lvalues    []*float64 // addresses of locations
	rvalues    []float64  // stored intermediate values
	elementals []uintptr  // elemental function indices
}

// A record specifies the record type and indexes the tape memory
// to specify record argument. At the cost of one redirection,
// the number of memory allocations is logarithmic in the number
// of instructions, and a record has a fixed size.
type record struct {
	typ, op int // record type and opcode or index of elemental
	il, ir  int // indices of the first pointer and value
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

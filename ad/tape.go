package ad

// Implementation of the AD tape

type Tape struct {
	records []record
	bars    map[*float64]float64
}

type record struct {
}

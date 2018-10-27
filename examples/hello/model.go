// "Hello World" model.
package hello

// y is the observation
type Model struct {
	y float64
}

// x[0] is the mean of the normal distribution
func (m *Model) Observe(x []float64) float64 {
    // d := x[0] - m.y
    d := 1.
	return -d * d
}

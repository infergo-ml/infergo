// "Hello World" model.
package hello

// y is the observation
type Model struct {
	y float64
}

// x[0] is the mean of the normal distribution
func (m *Model) Observe(x []float64) float64 {
    _d := x[0] - m.y
	return -_d*_d
}

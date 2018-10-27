// "Hello World" model.
package hello

// y is the observation
type Model struct {
	y float64
}

// x[0] is the mean of the normal distribution
func (m *Model) Observe(_ []float64) float64 {
	d := - m.y
	return -d * d
}

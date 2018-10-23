package hello

// Empty model
type Model struct {
}

func (m *Model) Observe(x []float64) float64 {
	return 0.
}

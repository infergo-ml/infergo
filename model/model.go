// Package model specifies the interface of a probabilistc
// model.
package model

// A probabilistic model must implement interface Model. Method
// Observe accepts a vector of parameters and returns the
// loglikelihood. Computation of the gradient is automatically
// induced through algorithmic differentiation.
type Model interface {
	Observe(parameters []float64) float64
}

// Singleton T of type t is used to define utility functions useful for
// implementing models.
type t struct {}
func (m t) Observe(x []float64) float64 {
	return 0
}
var T t

// Destructuring the parameters

// Shiftn shifts n elements from the slice pointed to by px into
// the slice pointed to by py.
func (m t) Shiftn(px *[]float64, n int, py *[]float64) {
	*py, *px = (*px)[:n], (*px)[n:]
}

// Shiftn shifts a single value from the slice pointed to by px into
// the float64 variable pointed to by py.
func (m t) Shift(px *[]float64, py *float64) {
	*py, *px = (*px)[0], (*px)[1:]
}

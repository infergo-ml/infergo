// Package model specifies the interface of a probabilistc
// model.
package model

import (
	"bitbucket.org/dtolpin/infergo/ad"
)

// A probabilistic model must implement interface Model. Method
// Observe accepts a vector of parameters and returns the
// loglikelihood. Computation of the gradient is automatically
// induced through algorithmic differentiation.
type Model interface {
	Observe(parameters []float64) float64
}

// An elemental model uses a supplied gradient instead of
// automatic differentation.
type ElementalModel interface {
	Model
	Gradient() []float64
}

// Shift shifts n parameters from x, useful for destructuring
// the parameter vector.
func Shift(px *[]float64, n int) []float64 {
	var y []float64
	y, *px = (*px)[:n], (*px)[n:]
	return y
}

// Gradient automatically selects either supplied or automatic
// gradient
func Gradient(m Model) []float64 {
	switch m := m.(type) {
	case ElementalModel:
		return m.Gradient()
	default:
		return ad.Gradient()
	}
}

// Inference algorithms
package infer

import (
	"bitbucket.org/dtolpin/infergo/model"
	"bitbucket.org/dtolpin/infergo/ad"
)

// Optimizer is the interface of gradient-based
// optimizers.
type Optimizer interface {
	Advance(m model.Model, x []float64)
}

// Gradient descent (or ascent for negative step).
type GD struct {
	Rate float64
	Decay float64
}

func (opt *GD) Step(m model.Model, x []float64) {
	m.Observe(x)
	grad := ad.Gradient()
	for i := 0; i != len(x); i++ {
		x[i] -= grad[i] * opt.Rate
	}
	opt.Rate *= opt.Decay
}

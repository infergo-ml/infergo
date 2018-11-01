// Inference algorithms
package infer

import (
	"bitbucket.org/dtolpin/infergo/ad"
	"bitbucket.org/dtolpin/infergo/model"
)

// Optimizer is the interface of gradient-based
// optimizers.
type Optimizer interface {
	Advance(m model.Model, x []float64)
}

// Gradient ascent with momentum. If the momentum
// is not set, and thus 0., reduces to vanilla gradient
// ascent.
type Grad struct {
	Rate  float64    //learning rate
	Decay float64    // rate decay
    Momentum float64 // gradient momentum factor
    lastUpdate []float64
}

func (opt *Grad) Step(m model.Model, x []float64) {
	m.Observe(x)
	grad := ad.Gradient()
    if opt.lastUpdate == nil {
        // lastUpdate is initialized to zeros.
        opt.lastUpdate = make([]float64, len(x))
    }
	for i := 0; i != len(x); i++ {
        update := (1. - opt.Momentum) * opt.Rate * grad[i] +
            opt.lastUpdate[i] * opt.Momentum
		x[i] += update
        opt.lastUpdate[i] = update
	}
	opt.Rate *= opt.Decay
}

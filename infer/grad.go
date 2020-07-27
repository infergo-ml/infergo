package infer

// Gradient ascent algorithms.

import (
	"bitbucket.org/dtolpin/infergo/model"
	"math"
)

// Grad is the interface of gradient-based
// optimizers. Step makes a single step over
// parameters in the gradient direction.
type Grad interface {
	Step(m model.Model, x []float64) (ll float64, grad []float64)
}

// Gradient ascent with momentum
// (https://www.nature.com/articles/323533a0). If the momentum
// factor is not set, and thus 0, reduces to vanilla gradient
// ascent.
type Momentum struct {
	Rate  float64   //learning rate
	Decay float64   // rate decay
	Gamma float64   // gradient momentum factor
	u     []float64 // last update
}

// Step implements the Optimizer interface.
func (opt *Momentum) Step(
	m model.Model,
	x []float64,
) (
	ll float64,
	grad []float64,
) {
	ll, grad = m.Observe(x), model.Gradient(m)
	if opt.u == nil {
		// u is initialized to zeros.
		opt.u = make([]float64, len(x))
	}
	for i := range x {
		u := opt.Rate*grad[i] + opt.u[i]*opt.Gamma
		x[i] += u
		opt.u[i] = u
	}
	opt.Rate *= opt.Decay
	return ll, grad
}

// Adam (https://arxiv.org/abs/1412.6980).
type Adam struct {
	Rate  float64   // learning rate
	Beta1 float64   // first momentum factor
	Beta2 float64   // second momentum factor
	Eps   float64   // stabilizer
	u     []float64 // first momentum
	v     []float64 // second momentum
	b1t   float64   // Beta1^t
	b2t   float64   // Beta2^t
}

// Step implements the Optimizer interface.
func (opt *Adam) Step(
	m model.Model,
	x []float64,
) (
	ll float64,
	grad []float64,
) {
	ll, grad = m.Observe(x), model.Gradient(m)

	if opt.u == nil {
		opt.setDefaults()
		// The momenta are initalized to zeros.
		opt.u = make([]float64, len(x))
		opt.v = make([]float64, len(x))
		opt.b1t = opt.Beta1
		opt.b2t = opt.Beta2
	}

	for i := range x {
		// Compute the new momenta.
		u := opt.Beta1*opt.u[i] + (1-opt.Beta1)*grad[i]
		v := opt.Beta2*opt.v[i] + (1-opt.Beta2)*grad[i]*grad[i]
		opt.u[i] = u
		opt.v[i] = v

		// Correct the bias.
		u /= (1 - opt.b1t)
		v /= (1 - opt.b2t)

		// Update the parameters.
		x[i] += opt.Rate / (math.Sqrt(v) + opt.Eps) * u
	}

	// Update momentum factors for the next step.
	opt.b1t *= opt.Beta1
	opt.b2t *= opt.Beta2

	return ll, grad
}

// setDefaults sets default parameter values for the Adam
// optimizer unless initialized.
func (opt *Adam) setDefaults() {
	if opt.Beta1 == 0 {
		opt.Beta1 = 0.9
	}
	if opt.Beta2 == 0 {
		opt.Beta2 = 0.999
	}
	if opt.Eps == 0 {
		opt.Eps = 1e-8
	}
}

// Optimize wraps a gradient-based optimizer into
// an optimization loop with early stopping if a
// plateau is reached.
func Optimize(
	opt Grad,
	m model.Model, theta []float64, 
	niter, nplateau int,
	eps float64,
) (
	ll0, ll float64,
	iter int,
) {
	ll0, _ = opt.Step(m, theta)
	ll, llprev := ll0, ll0
	iter = 0
	plateau := 0
	// Evolve x_, keep x_ with the highest log-likelihood in x
	theta_ := make([]float64, len(theta))
	copy(theta_, theta)
	for ; iter != niter; iter++ {
		ll_, _ := opt.Step(m, theta_)
		// Store theta_ in theta if log-likelihood increased
		if ll_ > ll {
			copy(theta, theta_)
			ll = ll_
		}
		// Stop early if the optimization is stuck
		if ll_-llprev <= eps {
			plateau++
			if plateau == nplateau {
				break
			}
		} else {
			plateau = 0
		}
		llprev = ll_
	}

	return ll0, ll, iter
}

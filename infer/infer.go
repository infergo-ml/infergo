// Inference algorithms:
//   * maximum likelihood estimation: gradient ascent;
//   * approximation of the posterior: Markov Chain Monte Carlo.
package infer

import (
	"bitbucket.org/dtolpin/infergo/model"
)

// Optimizer is the interface of gradient-based
// optimizers. Step makes a single step over
// parameters in the gradient direction.
type Optimizer interface {
	Step(m model.Model, x []float64) (ll float64, grad []float64)
}

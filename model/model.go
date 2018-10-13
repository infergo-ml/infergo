// declarations for model
package model

// Model is a probabilistic model.
// Method Observe accepts a vector of parameters and returns
// the loglikelihood and the Jacobian of the loglikelihood
// w.r.t. the parameters.
type Model interface {
	Observe(parameters []float64) float64
}

// DiffModel is a differentiable probabilistic model.
// The programmer defines the Observe method of Model.
// Computation of the Jacobian is automatically induced through
// algorithmic differentiation.
type DiffModel interface {
	Model
	Jacobian() []float64
}

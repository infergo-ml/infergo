// declarations for model
package model

type Model interface {
	// Method Observe accepts a vector of parameters and returns
	// the loglikelihood and the Jacobian of the loglikelihood
	// w.r.t. the parameters.
	Observe(parameters []float64) (loglikelihood float64, jacobian []float64)
	// The programmer defines the method which computes the loglikelihood,
	// computation of the Jacobian is automatically induced through
	// algorithmic differentiation.
}

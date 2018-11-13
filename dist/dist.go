// Package dist contains differentiatable distribution models.
// The package is automatically differentiated by deriv during
// build. In models, package
//         "bitbucket.org/dtolpin/infergo/dist/ad"
// should be imported instead for normal use.
package dist

import (
	"math"
)

// Normal distribution.
type normal struct{}

// Normal distribution, singleton instance.
var Normal normal

// Observe implements the Model interface. The parameter
// vector is mean, logvariance, observations.
func (dist normal) Observe(x []float64) float64 {
	mu, logv, y := x[0], x[1], x[2:]
	if len(y) == 1 {
		return dist.Logp(mu, logv, y[0])
	} else {
		return dist.Logps(mu, logv, y...)
	}
}

var log2pi float64

func init() {
	log2pi = math.Log(2. * math.Pi)
}

// Logp computes the log pdf of a single observation.
func (_ normal) Logp(mu, logv float64, y float64) float64 {
	vari := math.Exp(logv)
	d := y - mu
	return -0.5 * (d*d/vari + logv + log2pi)
}

// Logp computes the log pdf of a vector of observations.
func (_ normal) Logps(mu, logv float64, y ...float64) float64 {
	vari := math.Exp(logv)
	ll := 0.
	for i := 0; i != len(y); i++ {
		d := y[i] - mu
		ll += -0.5 * (d*d/vari + logv + log2pi)
	}
	return ll
}

// Exponential distribution.
type expon struct{}

// Exponential distribution, singleton instance.
var Expon expon

// Observe implements the Model interface. The parameter
// vector is loglambda, observations.
func (dist expon) Observe(x []float64) float64 {
	logl, y := x[0], x[1:]
	if len(y) == 1 {
		return dist.Logp(logl, y[0])
	} else {
		return dist.Logps(logl, y...)
	}
}

// Logp computes the log pdf of a single observation.
func (_ expon) Logp(logl float64, y float64) float64 {
	lambda := math.Exp(logl)
	return logl - lambda*y
}

// Logp computes the log pdf of a vector of observations.
func (_ expon) Logps(logl float64, y ...float64) float64 {
	ll := 0.
	lambda := math.Exp(logl)
	for i := 0; i != len(y); i++ {
		ll += logl - lambda*y[i]
	}
	return ll
}

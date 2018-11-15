// Package dist contains differentiatable distribution models.
// The package is automatically differentiated by deriv during
// build.
package dist

import (
	"bitbucket.org/dtolpin/infergo/mathx"
	"math"
)

// Normal distribution.
type normal struct{}

// Normal distribution, singleton instance.
var Normal normal

// Observe implements the Model interface. The parameter
// vector is mu, sigma, observations.
func (dist normal) Observe(x []float64) float64 {
	mu, sigma, y := x[0], x[1], x[2:]
	if len(y) == 1 {
		return dist.Logp(mu, sigma, y[0])
	} else {
		return dist.Logps(mu, sigma, y...)
	}
}

var log2pi float64

func init() {
	log2pi = math.Log(2. * math.Pi)
}

// Logp computes the log pdf of a single observation.
func (_ normal) Logp(mu, sigma float64, y float64) float64 {
	vari := sigma * sigma
	logv := math.Log(vari)
	d := y - mu
	return -0.5 * (d*d/vari + logv + log2pi)
}

// Logp computes the log pdf of a vector of observations.
func (_ normal) Logps(mu, sigma float64, y ...float64) float64 {
	vari := sigma * sigma
	logv := math.Log(vari)
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
// vector is lambda, observations.
func (dist expon) Observe(x []float64) float64 {
	lambda, y := x[0], x[1:]
	if len(y) == 1 {
		return dist.Logp(lambda, y[0])
	} else {
		return dist.Logps(lambda, y...)
	}
}

// Logp computes the log pdf of a single observation.
func (_ expon) Logp(lambda float64, y float64) float64 {
	logl := math.Log(lambda)
	return logl - lambda*y
}

// Logp computes the log pdf of a vector of observations.
func (_ expon) Logps(lambda float64, y ...float64) float64 {
	ll := 0.
	logl := math.Log(lambda)
	for i := 0; i != len(y); i++ {
		ll += logl - lambda*y[i]
	}
	return ll
}

// Gamma distribution
type gamma struct{}

// Gamma distribution, singleton instance.
var Gamma gamma

// Observe implements the Model interface. The parameter
// vector is alpha, beta, observations.
func (dist gamma) Observe(x []float64) float64 {
	alpha, beta, y := x[0], x[1], x[2:]
	if len(y) == 1 {
		return dist.Logp(alpha, beta, y[0])
	} else {
		return dist.Logps(alpha, beta, y...)
	}
}

// Logp computes the log pdf of a single observation.
func (_ gamma) Logp(alpha, beta float64, y float64) float64 {
	return (alpha-1)*math.Log(y) - beta*y -
		mathx.LogGamma(alpha) + alpha*math.Log(beta)
}

// Logp computes the log pdf of a vector of observations.
func (_ gamma) Logps(alpha, beta float64, y ...float64) float64 {
	ll := 0.
	for i := 0; i != len(y); i++ {
		ll += (alpha-1)*math.Log(y[i]) - beta*y[i] -
			mathx.LogGamma(alpha) + alpha*math.Log(beta)
	}
	return ll
}

// Beta distribution
type beta struct{}

// Beta distribution, singleton instance.
var Beta beta

// Observe implements the Model interface. The parameter
// vector is alpha, beta, observations.
func (dist beta) Observe(x []float64) float64 {
	alpha, beta, y := x[0], x[1], x[2:]
	if len(y) == 1 {
		return dist.Logp(alpha, beta, y[0])
	} else {
		return dist.Logps(alpha, beta, y...)
	}
}

// Logp computes the log pdf of a single observation.
func (_ beta) Logp(alpha, beta float64, y float64) float64 {
	return (alpha-1)*math.Log(y) +
		(beta-1)*math.Log(1-y) -
		mathx.LogGamma(alpha) - mathx.LogGamma(beta) +
		mathx.LogGamma(alpha+beta)
}

// Logp computes the log pdf of alpha vector of observations.
func (_ beta) Logps(alpha, beta float64, y ...float64) float64 {
	ll := 0.
	for i := 0; i != len(y); i++ {
		ll += (alpha-1)*math.Log(y[i]) +
			(beta-1)*math.Log(1-y[i]) -
			mathx.LogGamma(alpha) - mathx.LogGamma(beta) +
			mathx.LogGamma(alpha+beta)
	}
	return ll
}

// Package dist contains differentiatable distribution models.
// The package is automatically differentiated by deriv during
// build. In models, package
//         "bitbucket.org/dtolpin/infergo/dist/ad"
// should be imported instead for normal use.
package dist

import (
	"math"
)

// Normal distribution
type normal struct{}

var Normal normal

func (dist normal) Observe(x []float64) float64 {
	mu, logv, y := x[0], x[1], x[2:]
	return dist.Logp(mu, logv, y...)
}

var log2pi float64

func init() {
	log2pi = math.Log(2. * math.Pi)
}

func (_ normal) Logp(mu, logv float64, y ...float64) float64 {
	ll := 0.
	for i := 0; i != len(y); i++ {
		d := y[i] - mu
		vari := math.Exp(logv)
		ll += -0.5 * (d*d/vari + logv + log2pi)
	}
	return ll
}

// Exponential distribution
type expon struct{}

var Expon expon

func (dist expon) Observe(x []float64) float64 {
	lambda, y := x[0], x[1:]
	return dist.Logp(lambda, y...)
}

func (_ expon) Logp(lambda float64, y ...float64) float64 {
	ll := 0.
	for i := 0; i != len(y); i++ {
		ll += math.Log(lambda) - lambda*y[i]
	}
	return ll
}

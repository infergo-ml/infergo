// Package dist contains differentiatable distribution models.
package dist

import (
	"math"
)

// Normal distribution
type normal struct{}

var Normal normal

func (dist normal) Observe(x []float64) float64 {
	y, mu, logv := x[0], x[1], x[2]
	return dist.Logp(y, mu, logv)
}

var log2pi float64

func init() {
	log2pi = math.Log(2. * math.Pi)
}

func (_ normal) Logp(y, mu, logv float64) float64 {
	d := y - mu
	vari := math.Exp(logv)
	return -0.5 * (d*d/vari + logv + log2pi)
}

// Exponential distribution
type expon struct{}

var Expon expon

func (dist expon) Observe(x []float64) float64 {
	y, lambda := x[0], x[1]
	return dist.Logp(y, lambda)
}

func (_ expon) Logp(y, lambda float64) float64 {
	return math.Log(lambda) - lambda*y
}

// Package dist contains differentiatable distribution models.
package dist

import (
	"bitbucket.org/dtolpin/infergo/model"
	"math"
)

// Dist is the interface of a distribution
type Dist interface {
	model.Model
	Logp(x ...float64) float64
}

// Normal distribution
type NormalDist struct{}

var Normal NormalDist

func (dist NormalDist) Observe(x []float64) float64 {
	return dist.Logp(x...)
}

var log2pi float64

func init() {
	log2pi = math.Log(2. * math.Pi)
}

func (_ NormalDist) Logp(x ...float64) float64 {
	y, mu, logv := x[0], x[1], x[2]
	d := y - mu
	vari := math.Exp(logv)
	return -0.5 * (d*d/vari + logv + log2pi)
}

// Exponential distribution
type ExponDist struct{}

var Expon ExponDist

func (dist ExponDist) Observe(x []float64) float64 {
	return dist.Logp(x...)
}

func (_ ExponDist) Logp(x ...float64) float64 {
	y, lambda := x[0], x[1]
	return math.Log(lambda) - lambda*y
}

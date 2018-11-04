// Package dist contains differentiatable distribution models.
package dist

import (
	"math"
)

// Normal distribution
type normal struct {}
var Normal normal


func (dist normal) Observe(x []float64) float64 {
	return dist.Pdf(x[0], x[1], x[2])
}

var log2pi float64
func init() {
	log2pi = math.Log(2. * math.Pi)
}

func (_ normal) Pdf(x, mu, logv float64) float64 {
	d := x - mu
	vari := math.Exp(logv)
	return -0.5*(d*d/vari + logv + log2pi)
}

// Exponential distribution
type expon struct {}
var Expon expon

func (dist expon) Observe(x []float64) float64 {
	return dist.Pdf(x[0], x[1])
}

func (_ expon) Pdf(x, lambda float64) float64 {
	return math.Log(lambda) - lambda * x
}

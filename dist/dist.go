// Package dist contains differentiatable distribution models.
package dist

import (
	"math"
)

// Normal distribution
type Normal struct {
	X float64
}

func (dist *Normal) Observe(x []float64) float64 {
	return dist.Pdf(x[0], x[1])
}

var log2pi float64
func init() {
	log2pi = math.Log(2. * math.Pi)
}

func (dist *Normal) Pdf(mu, logv float64) float64 {
	d := dist.X - mu
	vari := math.Exp(logv)
	return -0.5*(d*d/vari + logv + log2pi)
}

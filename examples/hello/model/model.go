// Inferring parameters of the Normal distribution from
// observations
package model

import (
	. "bitbucket.org/dtolpin/infergo/dist/ad"
    "math"
)

// data are the observations
type Model struct {
	Data []float64
}

// x[0] is the mean, x[1] is the logvariance of the distribution
func (m *Model) Observe(x []float64) float64 {
	return Normal.Logps(x[0], math.Exp(x[1]), m.Data...)
}

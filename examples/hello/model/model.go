// Inferring parameters of the Normal distribution from
// observations
package model

import (
	. "bitbucket.org/dtolpin/infergo/dist"
	"math"
)

// data are the observations
type Model struct {
	Data []float64
}

// x[0] is the mean, x[1] is the log stddev of the distribution
func (m *Model) Observe(x []float64) float64 {
	// Our prior is a unit normal ...
	ll := Normal.Logps(0, 1, x...)
	// ... but the posterior is based on data observations.
	ll += Normal.Logps(x[0], math.Exp(x[1]), m.Data...)
	return ll
}

// Inferring parameters of the Normal distribution from
// observations
package model

import (
	. "bitbucket.org/dtolpin/infergo/dist/ad"
)

// data are the observations
type Model struct {
	Data []float64
}

// x[0] is the mean, x[1] is the logvariance of the distribution
func (m *Model) Observe(x []float64) float64 {
	ll := 0.
	for i := 0; i != len(m.Data); i++ {
		ll += Normal.Logp(m.Data[i], x[0], x[1])
	}
	return ll
}

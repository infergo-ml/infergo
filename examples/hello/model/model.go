// Inferring parameters of the Normal distribution from
// observations
package model

import (
	dist "bitbucket.org/dtolpin/infergo/dist/ad"
	"math"
)

// data are the observations
type Model struct {
	Data []float64
}

// x[0] is the mean, x[1] is the logvariance of the distribution
func (m *Model) Observe(x []float64) float64 {
	mean := x[0]
	logv := x[1]
	vari := math.Exp(logv)
	ll := 0.
	for i := 0; i != len(m.Data); i++ {
		ll += dist.Normal{m.Data[i]}.Pdf(mean, logv)
	}
	return ll
}

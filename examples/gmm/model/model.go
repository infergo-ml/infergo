// Gaussian mixture
package model

import (
	. "bitbucket.org/dtolpin/infergo/dist"
	"bitbucket.org/dtolpin/infergo/mathx"
	"math"
)

// data are the observations
type Model struct {
	Data  []float64 // samples
	NComp int       // number of components
}

func (m *Model) Observe(x []float64) float64 {
	ll := 0.0
	mu := make([]float64, m.NComp)
	sigma := make([]float64, m.NComp)

	// Fetch component parameters
	for j := 0; j != m.NComp; j++ {
		mu[j] = x[2*j]
		sigma[j] = math.Exp(x[2*j+1])
	}

	// Compute log likelihood of mixture
	// given the data
	for i := 0; i != len(m.Data); i++ {
		var l float64
		for j := 0; j != m.NComp; j++ {
			lj := Normal.Logp(mu[j], sigma[j], m.Data[i])
			if j == 0 {
				l = lj
			} else {
				l = mathx.LogSumExp(l, lj)
			}
		}
		ll += l
	}
	return ll
}

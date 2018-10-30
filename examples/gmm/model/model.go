// Gaussian mixture
package normal

import (
	"math"
	"bitbucket.org/dtolpin/infergo/mathx"
)

// data are the observations
type Model struct {
	Data []float64 // samples
	NComp int      // number of components
}

func (m *Model) Observe(x []float64) float64 {
	mean := make([]float64, m.NComp)
	logv := make([]float64, m.NComp)
	vari := make([]float64, m.NComp)
	
	// Fetch component parameters
	for j := 0; j != m.NComp; j++ {
		mean[j] = x[2*j]
		logv[j] = x[2*j + 1]
		vari[j] = math.Exp(logv[j])
	}

	// Compute log likelihood of mixture 
	// given the data
	ll := 0.0
	for i := 0; i != len(m.Data); i++ {
		var l float64
		for  j := 0; j != m.NComp; j++ {
			d := m.Data[i] - mean[j]
			lj := -d*d/vari[j] - logv[j]
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

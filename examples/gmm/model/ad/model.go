package model

import (
	"bitbucket.org/dtolpin/infergo/ad"
	. "bitbucket.org/dtolpin/infergo/dist/ad"
	"bitbucket.org/dtolpin/infergo/mathx"
	"math"
)

type Model struct {
	Data  []float64
	NComp int
}

func (m *Model) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	var ll float64
	ad.Assignment(&ll, ad.Value(0.0))
	var mu []float64

	mu = make([]float64, m.NComp)
	var sigma []float64

	sigma = make([]float64, m.NComp)

	for j := 0; j != m.NComp; j = j + 1 {
		ad.Assignment(&mu[j], &x[2*j])
		ad.Assignment(&sigma[j], ad.Elemental(math.Exp, &x[2*j+1]))
	}

	for i := 0; i != len(m.Data); i = i + 1 {
		var l float64
		for j := 0; j != m.NComp; j = j + 1 {
			var lj float64
			ad.Assignment(&lj, ad.Call(func(_vararg []float64) {
				Normal.Logp(0, 0, 0)
			}, 3, &mu[j], &sigma[j], &m.Data[i]))
			if j == 0 {
				ad.Assignment(&l, &lj)
			} else {
				ad.Assignment(&l, ad.Elemental(mathx.LogSumExp, &l, &lj))
			}
		}
		ad.Assignment(&ll, ad.Arithmetic(ad.OpAdd, &ll, &l))
	}
	return ad.Return(&ll)
}

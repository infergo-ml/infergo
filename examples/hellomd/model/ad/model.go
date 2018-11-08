package model

import (
	"bitbucket.org/dtolpin/infergo/ad"
	. "bitbucket.org/dtolpin/infergo/dist/ad"
)

type Model struct {
	Data []float64
}

func (m *Model) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	var ll float64
	ad.Assignment(&ll, ad.Value(0.))
	var N int

	N = len(x) / 2
	for j := 0; j != N; j = j + 1 {
		for i := 0; i != len(m.Data); i = i + 1 {
			ad.Assignment(&ll, ad.Arithmetic(ad.OpAdd, &ll, ad.Call(func(_vararg []float64) {
				Normal.Logp(0, 0, 0)
			}, 3, &m.Data[i], &x[2*j], &x[2*j+1])))
		}
	}
	return ad.Return(&ll)
}

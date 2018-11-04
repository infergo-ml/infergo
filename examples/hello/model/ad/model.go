package model

import (
	"bitbucket.org/dtolpin/infergo/dist/ad"
	"bitbucket.org/dtolpin/infergo/ad"
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
	for i := 0; i != len(m.Data); i = i + 1 {
		ad.Assignment(&ll, ad.Arithmetic(ad.OpAdd, &ll, ad.Call(func(_vararg []float64) {
			dist.Normal{m.Data[i]}.Observe(x)
		}, 0)))
	}
	return ad.Return(&ll)
}

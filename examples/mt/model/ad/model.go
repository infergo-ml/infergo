package model

import (
	"bitbucket.org/dtolpin/infergo/ad"
	. "bitbucket.org/dtolpin/infergo/dist/ad"
	"math"
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
	ad.Assignment(&ll, ad.Call(func(_ []float64) {
		Normal.Logps(0, 0, x...)
	}, 2, ad.Value(0), ad.Value(1)))
	ad.Assignment(&ll, ad.Arithmetic(ad.OpAdd, &ll, ad.Call(func(_ []float64) {
		Normal.Logps(0, 0, m.Data...)
	}, 2, &x[0], ad.Elemental(math.Exp, &x[1]))))
	return ad.Return(&ll)
}

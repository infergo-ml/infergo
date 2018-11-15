package model

import (
	"bitbucket.org/dtolpin/infergo/ad"
	. "bitbucket.org/dtolpin/infergo/dist/ad"
	"math"
)

type Model struct {
	J          int
	Y          []float64
	Sigma      []float64
	Stau, Seta float64
}

func (m *Model) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	var mu float64
	ad.Assignment(&mu, &x[0])
	var tau float64
	ad.Assignment(&tau, ad.Elemental(math.Exp, &x[1]))
	var eta []float64

	eta = x[2:]
	var ll float64
	ad.Assignment(&ll, ad.Call(func(_vararg []float64) {
		Normal.Logp(0, 0, 0)
	}, 3, ad.Value(0), &m.Stau, &x[1]))
	ad.Assignment(&ll, ad.Arithmetic(ad.OpAdd, &ll, ad.Call(func(_vararg []float64) {
		Normal.Logps(0, 0, eta...)
	}, 2, ad.Value(0), &m.Seta)))
	for i, y := range m.Y {
		var theta float64
		ad.Assignment(&theta, ad.Arithmetic(ad.OpAdd, &mu, ad.Arithmetic(ad.OpMul, &tau, &eta[i])))
		ad.Assignment(&ll, ad.Arithmetic(ad.OpAdd, &ll, ad.Call(func(_vararg []float64) {
			Normal.Logp(0, 0, 0)
		}, 3, &theta, &m.Sigma[i], &y)))
	}
	return ad.Return(&ll)
}

package model

import (
	. "bitbucket.org/dtolpin/infergo/dist/ad"
	"bitbucket.org/dtolpin/infergo/ad"
	"math"
)

type Model struct {
	J			int
	Y			[]float64
	Sigma			[]float64
	LogVtau, LogVeta	float64
}

func (m *Model) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	var ll float64
	ad.Assignment(&ll, ad.Value(0.0))
	var mu float64
	ad.Assignment(&mu, &x[0])
	ad.Assignment(&ll, ad.Arithmetic(ad.OpAdd, &ll, ad.Call(func(_vararg []float64) {
		Normal.Pdf(0, 0, 0)
	}, 3, &x[1], ad.Value(0), &m.LogVtau)))
	var tau float64
	ad.Assignment(&tau, ad.Elemental(math.Exp, &x[1]))
	var eta []float64

	eta = x[2:]

	for i, y := range m.Y {
		ad.Assignment(&ll, ad.Arithmetic(ad.OpAdd, &ll, ad.Call(func(_vararg []float64) {
			Normal.Pdf(0, 0, 0)
		}, 3, &eta[i], ad.Value(0), &m.LogVeta)))
		var theta float64
		ad.Assignment(&theta, ad.Arithmetic(ad.OpAdd, &mu, ad.Arithmetic(ad.OpMul, &tau, &eta[i])))
		var logVtheta float64
		ad.Assignment(&logVtheta, ad.Elemental(math.Log, ad.Arithmetic(ad.OpMul, &m.Sigma[i], &m.Sigma[i])))
		ad.Assignment(&ll, ad.Arithmetic(ad.OpAdd, &ll, ad.Call(func(_vararg []float64) {
			Normal.Pdf(0, 0, 0)
		}, 3, &y, &theta, &logVtheta)))
	}
	return ad.Return(&ll)
}

package model

import (
	"math"
	"bitbucket.org/dtolpin/infergo/ad"
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
	ad.Assignment(&ll, ad.Arithmetic(ad.OpSub, &ll, ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpDiv, ad.Arithmetic(ad.OpMul, &x[1], &x[1]), ad.Elemental(math.Exp, &m.LogVtau)), &m.LogVtau)))
	var tau float64
	ad.Assignment(&tau, ad.Elemental(math.Exp, &x[1]))
	var eta []float64

	eta = x[2:]

	for i, y := range m.Y {
		ad.Assignment(&ll, ad.Arithmetic(ad.OpSub, &ll, ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpDiv, ad.Arithmetic(ad.OpMul, &eta[i], &eta[i]), ad.Elemental(math.Exp, &m.LogVeta)), &m.LogVeta)))
		var theta float64
		ad.Assignment(&theta, ad.Arithmetic(ad.OpAdd, &mu, ad.Arithmetic(ad.OpMul, &tau, &eta[i])))
		var sigma2 float64
		ad.Assignment(&sigma2, ad.Arithmetic(ad.OpMul, &m.Sigma[i], &m.Sigma[i]))
		var d float64
		ad.Assignment(&d, ad.Arithmetic(ad.OpSub, &y, &theta))
		ad.Assignment(&ll, ad.Arithmetic(ad.OpSub, &ll, ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpDiv, ad.Arithmetic(ad.OpMul, &d, &d), &sigma2), ad.Elemental(math.Log, &sigma2))))
	}
	return ad.Return(&ll)
}

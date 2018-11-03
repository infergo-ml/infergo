package model

import (
	"math"
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
	var mean float64
	ad.Assignment(&mean, &x[0])
	var logv float64
	ad.Assignment(&logv, &x[1])
	var vari float64
	ad.Assignment(&vari, ad.Elemental(math.Exp, &logv))
	var ll float64
	ad.Assignment(&ll, ad.Value(0.))
	for i := 0; i != len(m.Data); i = i + 1 {
		var d float64
		ad.Assignment(&d, ad.Arithmetic(ad.OpSub, &m.Data[i], &mean))
		ad.Assignment(&ll, ad.Arithmetic(ad.OpSub, &ll, ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpDiv, ad.Arithmetic(ad.OpMul, &d, &d), &vari), &logv)))
	}
	return ad.Return(&ll)
}

package normal

import (
	"bitbucket.org/dtolpin/infergo/mathx"
	"bitbucket.org/dtolpin/infergo/ad"
	"math"
)

type Model struct {
	Data	[]float64
	NComp	int
}

func (m *Model) Observe(x []float64) float64 {
	ad.Setup(x)
	var mean []float64

	mean = make([]float64, m.NComp)
	var logv []float64

	logv = make([]float64, m.NComp)
	var vari []float64

	vari = make([]float64, m.NComp)

	for j := 0; j != m.NComp; j = j + 1 {
		ad.Assignment(&mean[j], &x[2*j])
		ad.Assignment(&logv[j], &x[2*j+1])
		ad.Assignment(&vari[j], ad.Elemental(math.Exp, &logv[j]))
	}
	var ll float64
	ad.Assignment(&ll, ad.Value(0.0))
	for i := 0; i != len(m.Data); i = i + 1 {
		var l float64
		for j := 0; j != m.NComp; j = j + 1 {
			var d float64
			ad.Assignment(&d, ad.Arithmetic(ad.OpSub, &m.Data[i], &mean[j]))
			var lj float64
			ad.Assignment(&lj, ad.Arithmetic(ad.OpSub, ad.Arithmetic(ad.OpDiv, ad.Arithmetic(ad.OpMul, ad.Arithmetic(ad.OpNeg, &d), &d), &vari[j]), &logv[j]))
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

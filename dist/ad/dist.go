package dist

import (
	"math"
	"bitbucket.org/dtolpin/infergo/ad"
)

type Normal struct {
	X float64
}

func (dist Normal) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	return ad.Return(ad.Call(func(_vararg []float64) {
		dist.Pdf(0, 0)
	}, 2, &x[0], &x[1]))
}

var log2pi float64

func init() {
	log2pi = math.Log(2. * math.Pi)
}

func (dist Normal) Pdf(mu, logv float64) float64 {
	if ad.Called() {
		ad.Enter(&mu, &logv)
	} else {
		panic("Pdf called outside Observe.")
	}
	var d float64
	ad.Assignment(&d, ad.Arithmetic(ad.OpSub, &dist.X, &mu))
	var vari float64
	ad.Assignment(&vari, ad.Elemental(math.Exp, &logv))
	return ad.Return(ad.Arithmetic(ad.OpMul, ad.Arithmetic(ad.OpNeg, ad.Value(0.5)), (ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpDiv, ad.Arithmetic(ad.OpMul, &d, &d), &vari), &logv), &log2pi))))
}

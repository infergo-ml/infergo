package dist

import (
	"bitbucket.org/dtolpin/infergo/ad"
	"math"
)

type normal struct{}

var Normal normal

func (dist normal) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	return ad.Return(ad.Call(func(_vararg []float64) {
		dist.Pdf(0, 0, 0)
	}, 3, &x[0], &x[1], &x[2]))
}

var log2pi float64

func init() {
	log2pi = math.Log(2. * math.Pi)
}

func (_ normal) Pdf(x, mu, logv float64) float64 {
	if ad.Called() {
		ad.Enter(&x, &mu, &logv)
	} else {
		panic("Pdf called outside Observe.")
	}
	var d float64
	ad.Assignment(&d, ad.Arithmetic(ad.OpSub, &x, &mu))
	var vari float64
	ad.Assignment(&vari, ad.Elemental(math.Exp, &logv))
	return ad.Return(ad.Arithmetic(ad.OpMul, ad.Arithmetic(ad.OpNeg, ad.Value(0.5)), (ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpDiv, ad.Arithmetic(ad.OpMul, &d, &d), &vari), &logv), &log2pi))))
}

type expon struct{}

var Expon expon

func (dist expon) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	return ad.Return(ad.Call(func(_vararg []float64) {
		dist.Pdf(0, 0)
	}, 2, &x[0], &x[1]))
}

func (_ expon) Pdf(x, lambda float64) float64 {
	if ad.Called() {
		ad.Enter(&x, &lambda)
	} else {
		panic("Pdf called outside Observe.")
	}
	return ad.Return(ad.Arithmetic(ad.OpSub, ad.Elemental(math.Log, &lambda), ad.Arithmetic(ad.OpMul, &lambda, &x)))
}

package dist

import (
	"math"
	"bitbucket.org/dtolpin/infergo/ad"
)

type normal struct{}

var Normal normal

func (dist normal) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	var y float64
	var mu float64
	var logv float64
	ad.ParallelAssignment(&y, &mu, &logv, &x[0], &x[1], &x[2])
	return ad.Return(ad.Call(func(_vararg []float64) {
		dist.Logp(0, 0, 0)
	}, 3, &y, &mu, &logv))
}

var log2pi float64

func init() {
	log2pi = math.Log(2. * math.Pi)
}

func (_ normal) Logp(y, mu, logv float64) float64 {
	if ad.Called() {
		ad.Enter(&y, &mu, &logv)
	} else {
		panic("Logp called outside Observe.")
	}
	var d float64
	ad.Assignment(&d, ad.Arithmetic(ad.OpSub, &y, &mu))
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
	var y float64
	var lambda float64
	ad.ParallelAssignment(&y, &lambda, &x[0], &x[1])
	return ad.Return(ad.Call(func(_vararg []float64) {
		dist.Logp(0, 0)
	}, 2, &y, &lambda))
}

func (_ expon) Logp(y, lambda float64) float64 {
	if ad.Called() {
		ad.Enter(&y, &lambda)
	} else {
		panic("Logp called outside Observe.")
	}
	return ad.Return(ad.Arithmetic(ad.OpSub, ad.Elemental(math.Log, &lambda), ad.Arithmetic(ad.OpMul, &lambda, &y)))
}

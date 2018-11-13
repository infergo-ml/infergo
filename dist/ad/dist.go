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
	var mu float64
	var logv float64
	var y []float64

	mu, logv, y = x[0], x[1], x[2:]
	if len(y) == 1 {
		return ad.Return(ad.Call(func(_vararg []float64) {
			dist.Logp(0, 0, 0)
		}, 3, &mu, &logv, &y[0]))
	} else {
		return ad.Return(ad.Call(func(_vararg []float64) {
			dist.Logps(0, 0, y...)
		}, 2, &mu, &logv))
	}
}

var log2pi float64

func init() {
	log2pi = math.Log(2. * math.Pi)
}

func (_ normal) Logp(mu, logv float64, y float64) float64 {
	if ad.Called() {
		ad.Enter(&mu, &logv, &y)
	} else {
		panic("Logp called outside Observe.")
	}
	var vari float64
	ad.Assignment(&vari, ad.Elemental(math.Exp, &logv))
	var d float64
	ad.Assignment(&d, ad.Arithmetic(ad.OpSub, &y, &mu))
	return ad.Return(ad.Arithmetic(ad.OpMul, ad.Arithmetic(ad.OpNeg, ad.Value(0.5)), (ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpDiv, ad.Arithmetic(ad.OpMul, &d, &d), &vari), &logv), &log2pi))))
}

func (_ normal) Logps(mu, logv float64, y ...float64) float64 {
	if ad.Called() {
		ad.Enter(&mu, &logv)
	} else {
		panic("Logps called outside Observe.")
	}
	var vari float64
	ad.Assignment(&vari, ad.Elemental(math.Exp, &logv))
	var ll float64
	ad.Assignment(&ll, ad.Value(0.))
	for i := 0; i != len(y); i = i + 1 {
		var d float64
		ad.Assignment(&d, ad.Arithmetic(ad.OpSub, &y[i], &mu))
		ad.Assignment(&ll, ad.Arithmetic(ad.OpAdd, &ll, ad.Arithmetic(ad.OpMul, ad.Arithmetic(ad.OpNeg, ad.Value(0.5)), (ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpDiv, ad.Arithmetic(ad.OpMul, &d, &d), &vari), &logv), &log2pi)))))
	}
	return ad.Return(&ll)
}

type expon struct{}

var Expon expon

func (dist expon) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	var logl float64
	var y []float64

	logl, y = x[0], x[1:]
	if len(y) == 1 {
		return ad.Return(ad.Call(func(_vararg []float64) {
			dist.Logp(0, 0)
		}, 2, &logl, &y[0]))
	} else {
		return ad.Return(ad.Call(func(_vararg []float64) {
			dist.Logps(0, y...)
		}, 1, &logl))
	}
}

func (_ expon) Logp(logl float64, y float64) float64 {
	if ad.Called() {
		ad.Enter(&logl, &y)
	} else {
		panic("Logp called outside Observe.")
	}
	var lambda float64
	ad.Assignment(&lambda, ad.Elemental(math.Exp, &logl))
	return ad.Return(ad.Arithmetic(ad.OpSub, &logl, ad.Arithmetic(ad.OpMul, &lambda, &y)))
}

func (_ expon) Logps(logl float64, y ...float64) float64 {
	if ad.Called() {
		ad.Enter(&logl)
	} else {
		panic("Logps called outside Observe.")
	}
	var ll float64
	ad.Assignment(&ll, ad.Value(0.))
	var lambda float64
	ad.Assignment(&lambda, ad.Elemental(math.Exp, &logl))
	for i := 0; i != len(y); i = i + 1 {
		ad.Assignment(&ll, ad.Arithmetic(ad.OpAdd, &ll, ad.Arithmetic(ad.OpSub, &logl, ad.Arithmetic(ad.OpMul, &lambda, &y[i]))))
	}
	return ad.Return(&ll)
}

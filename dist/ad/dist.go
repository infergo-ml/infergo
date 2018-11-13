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
	var mu float64
	var logv float64
	var y []float64

	mu, logv, y = x[0], x[1], x[2:]
	return ad.Return(ad.Call(func(_vararg []float64) {
		dist.Logp(0, 0, y...)
	}, 2, &mu, &logv))
}

var log2pi float64

func init() {
	log2pi = math.Log(2. * math.Pi)
}

func (_ normal) Logp(mu, logv float64, y ...float64) float64 {
	if ad.Called() {
		ad.Enter(&mu, &logv)
	} else {
		panic("Logp called outside Observe.")
	}
	var ll float64
	ad.Assignment(&ll, ad.Value(0.))
	for i := 0; i != len(y); i = i + 1 {
		var d float64
		ad.Assignment(&d, ad.Arithmetic(ad.OpSub, &y[i], &mu))
		var vari float64
		ad.Assignment(&vari, ad.Elemental(math.Exp, &logv))
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
	var lambda float64
	var y []float64

	lambda, y = x[0], x[1:]
	return ad.Return(ad.Call(func(_vararg []float64) {
		dist.Logp(0, y...)
	}, 1, &lambda))
}

func (_ expon) Logp(lambda float64, y ...float64) float64 {
	if ad.Called() {
		ad.Enter(&lambda)
	} else {
		panic("Logp called outside Observe.")
	}
	var ll float64
	ad.Assignment(&ll, ad.Value(0.))
	for i := 0; i != len(y); i = i + 1 {
		ad.Assignment(&ll, ad.Arithmetic(ad.OpAdd, &ll, ad.Arithmetic(ad.OpSub, ad.Elemental(math.Log, &lambda), ad.Arithmetic(ad.OpMul, &lambda, &y[i]))))
	}
	return ad.Return(&ll)
}

package dist

import (
	"bitbucket.org/dtolpin/infergo/ad"
	"bitbucket.org/dtolpin/infergo/mathx"
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
	var sigma float64
	var y []float64

	mu, sigma, y = x[0], x[1], x[2:]
	if len(y) == 1 {
		return ad.Return(ad.Call(func(_vararg []float64) {
			dist.Logp(0, 0, 0)
		}, 3, &mu, &sigma, &y[0]))
	} else {
		return ad.Return(ad.Call(func(_vararg []float64) {
			dist.Logps(0, 0, y...)
		}, 2, &mu, &sigma))
	}
}

var log2pi float64

func init() {
	log2pi = math.Log(2. * math.Pi)
}

func (_ normal) Logp(mu, sigma float64, y float64) float64 {
	if ad.Called() {
		ad.Enter(&mu, &sigma, &y)
	} else {
		panic("Logp called outside Observe.")
	}
	var vari float64
	ad.Assignment(&vari, ad.Arithmetic(ad.OpMul, &sigma, &sigma))
	var logv float64
	ad.Assignment(&logv, ad.Elemental(math.Log, &vari))
	var d float64
	ad.Assignment(&d, ad.Arithmetic(ad.OpSub, &y, &mu))
	return ad.Return(ad.Arithmetic(ad.OpMul, ad.Arithmetic(ad.OpNeg, ad.Value(0.5)), (ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpDiv, ad.Arithmetic(ad.OpMul, &d, &d), &vari), &logv), &log2pi))))
}

func (_ normal) Logps(mu, sigma float64, y ...float64) float64 {
	if ad.Called() {
		ad.Enter(&mu, &sigma)
	} else {
		panic("Logps called outside Observe.")
	}
	var vari float64
	ad.Assignment(&vari, ad.Arithmetic(ad.OpMul, &sigma, &sigma))
	var logv float64
	ad.Assignment(&logv, ad.Elemental(math.Log, &vari))
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
	var lambda float64
	var y []float64

	lambda, y = x[0], x[1:]
	if len(y) == 1 {
		return ad.Return(ad.Call(func(_vararg []float64) {
			dist.Logp(0, 0)
		}, 2, &lambda, &y[0]))
	} else {
		return ad.Return(ad.Call(func(_vararg []float64) {
			dist.Logps(0, y...)
		}, 1, &lambda))
	}
}

func (_ expon) Logp(lambda float64, y float64) float64 {
	if ad.Called() {
		ad.Enter(&lambda, &y)
	} else {
		panic("Logp called outside Observe.")
	}
	var logl float64
	ad.Assignment(&logl, ad.Elemental(math.Log, &lambda))
	return ad.Return(ad.Arithmetic(ad.OpSub, &logl, ad.Arithmetic(ad.OpMul, &lambda, &y)))
}

func (_ expon) Logps(lambda float64, y ...float64) float64 {
	if ad.Called() {
		ad.Enter(&lambda)
	} else {
		panic("Logps called outside Observe.")
	}
	var ll float64
	ad.Assignment(&ll, ad.Value(0.))
	var logl float64
	ad.Assignment(&logl, ad.Elemental(math.Log, &lambda))
	for i := 0; i != len(y); i = i + 1 {
		ad.Assignment(&ll, ad.Arithmetic(ad.OpAdd, &ll, ad.Arithmetic(ad.OpSub, &logl, ad.Arithmetic(ad.OpMul, &lambda, &y[i]))))
	}
	return ad.Return(&ll)
}

type gamma struct{}

var Gamma gamma

func (dist gamma) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	var alpha float64
	var beta float64
	var y []float64

	alpha, beta, y = x[0], x[1], x[2:]
	if len(y) == 1 {
		return ad.Return(ad.Call(func(_vararg []float64) {
			dist.Logp(0, 0, 0)
		}, 3, &alpha, &beta, &y[0]))
	} else {
		return ad.Return(ad.Call(func(_vararg []float64) {
			dist.Logps(0, 0, y...)
		}, 2, &alpha, &beta))
	}
}

func (_ gamma) Logp(alpha, beta float64, y float64) float64 {
	if ad.Called() {
		ad.Enter(&alpha, &beta, &y)
	} else {
		panic("Logp called outside Observe.")
	}
	return ad.Return(ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpSub, ad.Arithmetic(ad.OpSub, ad.Arithmetic(ad.OpMul, (ad.Arithmetic(ad.OpSub, &alpha, ad.Value(1))), ad.Elemental(math.Log, &y)), ad.Arithmetic(ad.OpMul, &beta, &y)), ad.Elemental(mathx.LogGamma, &alpha)), ad.Arithmetic(ad.OpMul, &alpha, ad.Elemental(math.Log, &beta))))
}

func (_ gamma) Logps(alpha, beta float64, y ...float64) float64 {
	if ad.Called() {
		ad.Enter(&alpha, &beta)
	} else {
		panic("Logps called outside Observe.")
	}
	var ll float64
	ad.Assignment(&ll, ad.Value(0.))
	for i := 0; i != len(y); i = i + 1 {
		ad.Assignment(&ll, ad.Arithmetic(ad.OpAdd, &ll, ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpSub, ad.Arithmetic(ad.OpSub, ad.Arithmetic(ad.OpMul, (ad.Arithmetic(ad.OpSub, &alpha, ad.Value(1))), ad.Elemental(math.Log, &y[i])), ad.Arithmetic(ad.OpMul, &beta, &y[i])), ad.Elemental(mathx.LogGamma, &alpha)), ad.Arithmetic(ad.OpMul, &alpha, ad.Elemental(math.Log, &beta)))))
	}
	return ad.Return(&ll)
}

type beta struct{}

var Beta beta

func (dist beta) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	var alpha float64
	var beta float64
	var y []float64

	alpha, beta, y = x[0], x[1], x[2:]
	if len(y) == 1 {
		return ad.Return(ad.Call(func(_vararg []float64) {
			dist.Logp(0, 0, 0)
		}, 3, &alpha, &beta, &y[0]))
	} else {
		return ad.Return(ad.Call(func(_vararg []float64) {
			dist.Logps(0, 0, y...)
		}, 2, &alpha, &beta))
	}
}

func (_ beta) Logp(alpha, beta float64, y float64) float64 {
	if ad.Called() {
		ad.Enter(&alpha, &beta, &y)
	} else {
		panic("Logp called outside Observe.")
	}
	return ad.Return(ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpSub, ad.Arithmetic(ad.OpSub, ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpMul, (ad.Arithmetic(ad.OpSub, &alpha, ad.Value(1))), ad.Elemental(math.Log, &y)), ad.Arithmetic(ad.OpMul, (ad.Arithmetic(ad.OpSub, &beta, ad.Value(1))), ad.Elemental(math.Log, ad.Arithmetic(ad.OpSub, ad.Value(1), &y)))), ad.Elemental(mathx.LogGamma, &alpha)), ad.Elemental(mathx.LogGamma, &beta)), ad.Elemental(mathx.LogGamma, ad.Arithmetic(ad.OpAdd, &alpha, &beta))))
}

func (_ beta) Logps(alpha, beta float64, y ...float64) float64 {
	if ad.Called() {
		ad.Enter(&alpha, &beta)
	} else {
		panic("Logps called outside Observe.")
	}
	var ll float64
	ad.Assignment(&ll, ad.Value(0.))
	for i := 0; i != len(y); i = i + 1 {
		ad.Assignment(&ll, ad.Arithmetic(ad.OpAdd, &ll, ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpSub, ad.Arithmetic(ad.OpSub, ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpMul, (ad.Arithmetic(ad.OpSub, &alpha, ad.Value(1))), ad.Elemental(math.Log, &y[i])), ad.Arithmetic(ad.OpMul, (ad.Arithmetic(ad.OpSub, &beta, ad.Value(1))), ad.Elemental(math.Log, ad.Arithmetic(ad.OpSub, ad.Value(1), &y[i])))), ad.Elemental(mathx.LogGamma, &alpha)), ad.Elemental(mathx.LogGamma, &beta)), ad.Elemental(mathx.LogGamma, ad.Arithmetic(ad.OpAdd, &alpha, &beta)))))
	}
	return ad.Return(&ll)
}

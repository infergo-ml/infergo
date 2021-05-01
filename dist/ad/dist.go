package dist

import (
	"bitbucket.org/dtolpin/infergo/mathx"
	"bitbucket.org/dtolpin/infergo/ad"
	"fmt"
	"math"
)

var (
	logpi, log2pi float64
)

func init() {
	log2 := math.Log(2)
	logpi = math.Log(math.Pi)
	log2pi = log2 + logpi
}

type normal struct{}

var Normal normal

func (dist normal) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	var (
		mu	float64

		sigma	float64

		y	[]float64
	)

	mu, sigma, y = x[0], x[1], x[2:]
	if len(y) == 1 {
		return ad.Return(ad.Call(func(_ []float64) {
			dist.Logp(0, 0, 0)
		}, 3, &mu, &sigma, &y[0]))
	} else {
		return ad.Return(ad.Call(func(_ []float64) {
			dist.Logps(0, 0, y...)
		}, 2, &mu, &sigma))
	}
}

func (normal) Logp(mu, sigma float64, y float64) float64 {
	if ad.Called() {
		ad.Enter(&mu, &sigma, &y)
	} else {
		panic("Logp called outside Observe")
	}
	var vari float64
	ad.Assignment(&vari, ad.Arithmetic(ad.OpMul, &sigma, &sigma))
	var logv float64
	ad.Assignment(&logv, ad.Elemental(math.Log, &vari))
	var d float64
	ad.Assignment(&d, ad.Arithmetic(ad.OpSub, &y, &mu))
	return ad.Return(ad.Arithmetic(ad.OpMul, ad.Value(-0.5), (ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpDiv, ad.Arithmetic(ad.OpMul, &d, &d), &vari), &logv), &log2pi))))
}

func (normal) Logps(mu, sigma float64, y ...float64) float64 {
	if ad.Called() {
		ad.Enter(&mu, &sigma)
	} else {
		panic("Logps called outside Observe")
	}
	var vari float64
	ad.Assignment(&vari, ad.Arithmetic(ad.OpMul, &sigma, &sigma))
	var logv float64
	ad.Assignment(&logv, ad.Elemental(math.Log, &vari))
	var lp float64
	ad.Assignment(&lp, ad.Arithmetic(ad.OpMul, ad.Arithmetic(ad.OpMul, ad.Value(-0.5), (ad.Arithmetic(ad.OpAdd, &logv, &log2pi))), ad.Value(float64(len(y)))))
	for i := range y {
		var d float64
		ad.Assignment(&d, ad.Arithmetic(ad.OpSub, &y[i], &mu))
		ad.Assignment(&lp, ad.Arithmetic(ad.OpSub, &lp, ad.Arithmetic(ad.OpDiv, ad.Arithmetic(ad.OpMul, ad.Arithmetic(ad.OpMul, ad.Value(0.5), &d), &d), &vari)))
	}
	return ad.Return(&lp)
}

type cauchy struct{}

var Cauchy cauchy

func (dist cauchy) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	var (
		mu	float64

		sigma	float64

		y	[]float64
	)

	mu, sigma, y = x[0], x[1], x[2:]
	if len(y) == 1 {
		return ad.Return(ad.Call(func(_ []float64) {
			dist.Logp(0, 0, 0)
		}, 3, &mu, &sigma, &y[0]))
	} else {
		return ad.Return(ad.Call(func(_ []float64) {
			dist.Logps(0, 0, y...)
		}, 2, &mu, &sigma))
	}
}

func (cauchy) Logp(x0, gamma float64, y float64) float64 {
	if ad.Called() {
		ad.Enter(&x0, &gamma, &y)
	} else {
		panic("Logp called outside Observe")
	}
	var logGamma float64
	ad.Assignment(&logGamma, ad.Elemental(math.Log, &gamma))
	var d float64
	ad.Assignment(&d, ad.Arithmetic(ad.OpDiv, (ad.Arithmetic(ad.OpSub, &y, &x0)), &gamma))
	return ad.Return(ad.Arithmetic(ad.OpSub, ad.Arithmetic(ad.OpSub, ad.Arithmetic(ad.OpNeg, &logGamma), &logpi), ad.Elemental(math.Log, ad.Arithmetic(ad.OpAdd, ad.Value(1), ad.Arithmetic(ad.OpMul, &d, &d)))))
}

func (cauchy) Logps(x0, gamma float64, y ...float64) float64 {
	if ad.Called() {
		ad.Enter(&x0, &gamma)
	} else {
		panic("Logps called outside Observe")
	}
	var logGamma float64
	ad.Assignment(&logGamma, ad.Elemental(math.Log, &gamma))
	var lp float64
	ad.Assignment(&lp, ad.Arithmetic(ad.OpMul, (ad.Arithmetic(ad.OpSub, ad.Arithmetic(ad.OpNeg, &logGamma), &logpi)), ad.Value(float64(len(y)))))
	for i := range y {
		var d float64
		ad.Assignment(&d, ad.Arithmetic(ad.OpDiv, (ad.Arithmetic(ad.OpSub, &y[i], &x0)), &gamma))
		ad.Assignment(&lp, ad.Arithmetic(ad.OpSub, &lp, ad.Elemental(math.Log, ad.Arithmetic(ad.OpAdd, ad.Value(1), ad.Arithmetic(ad.OpMul, &d, &d)))))
	}
	return ad.Return(&lp)
}

type exponential struct{}

var Exponential, Expon exponential

func (dist exponential) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	var (
		lambda	float64

		y	[]float64
	)

	lambda, y = x[0], x[1:]
	if len(y) == 1 {
		return ad.Return(ad.Call(func(_ []float64) {
			dist.Logp(0, 0)
		}, 2, &lambda, &y[0]))
	} else {
		return ad.Return(ad.Call(func(_ []float64) {
			dist.Logps(0, y...)
		}, 1, &lambda))
	}
}

func (exponential) Logp(lambda float64, y float64) float64 {
	if ad.Called() {
		ad.Enter(&lambda, &y)
	} else {
		panic("Logp called outside Observe")
	}
	var logl float64
	ad.Assignment(&logl, ad.Elemental(math.Log, &lambda))
	return ad.Return(ad.Arithmetic(ad.OpSub, &logl, ad.Arithmetic(ad.OpMul, &lambda, &y)))
}

func (exponential) Logps(lambda float64, y ...float64) float64 {
	if ad.Called() {
		ad.Enter(&lambda)
	} else {
		panic("Logps called outside Observe")
	}
	var logl float64
	ad.Assignment(&logl, ad.Elemental(math.Log, &lambda))
	var lp float64
	ad.Assignment(&lp, ad.Arithmetic(ad.OpMul, &logl, ad.Value(float64(len(y)))))
	for i := range y {
		ad.Assignment(&lp, ad.Arithmetic(ad.OpSub, &lp, ad.Arithmetic(ad.OpMul, &lambda, &y[i])))
	}
	return ad.Return(&lp)
}

type gamma struct{}

var Gamma gamma

func (dist gamma) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	var (
		alpha	float64

		beta	float64

		y	[]float64
	)

	alpha, beta, y = x[0], x[1], x[2:]
	if len(y) == 1 {
		return ad.Return(ad.Call(func(_ []float64) {
			dist.Logp(0, 0, 0)
		}, 3, &alpha, &beta, &y[0]))
	} else {
		return ad.Return(ad.Call(func(_ []float64) {
			dist.Logps(0, 0, y...)
		}, 2, &alpha, &beta))
	}
}

func (gamma) Logp(alpha, beta float64, y float64) float64 {
	if ad.Called() {
		ad.Enter(&alpha, &beta, &y)
	} else {
		panic("Logp called outside Observe")
	}
	return ad.Return(ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpSub, ad.Arithmetic(ad.OpSub, ad.Arithmetic(ad.OpMul, (ad.Arithmetic(ad.OpSub, &alpha, ad.Value(1))), ad.Elemental(math.Log, &y)), ad.Arithmetic(ad.OpMul, &beta, &y)), ad.Elemental(mathx.LogGamma, &alpha)), ad.Arithmetic(ad.OpMul, &alpha, ad.Elemental(math.Log, &beta))))
}

func (gamma) Logps(alpha, beta float64, y ...float64) float64 {
	if ad.Called() {
		ad.Enter(&alpha, &beta)
	} else {
		panic("Logps called outside Observe")
	}
	var lp float64
	ad.Assignment(&lp, ad.Arithmetic(ad.OpMul, (ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpNeg, ad.Elemental(mathx.LogGamma, &alpha)), ad.Arithmetic(ad.OpMul, &alpha, ad.Elemental(math.Log, &beta)))), ad.Value(float64(len(y)))))
	for i := range y {
		ad.Assignment(&lp, ad.Arithmetic(ad.OpAdd, &lp, ad.Arithmetic(ad.OpSub, ad.Arithmetic(ad.OpMul, (ad.Arithmetic(ad.OpSub, &alpha, ad.Value(1))), ad.Elemental(math.Log, &y[i])), ad.Arithmetic(ad.OpMul, &beta, &y[i]))))
	}
	return ad.Return(&lp)
}

type beta struct{}

var Beta beta

func (dist beta) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	var (
		alpha	float64

		beta	float64

		y	[]float64
	)

	alpha, beta, y = x[0], x[1], x[2:]
	if len(y) == 1 {
		return ad.Return(ad.Call(func(_ []float64) {
			dist.Logp(0, 0, 0)
		}, 3, &alpha, &beta, &y[0]))
	} else {
		return ad.Return(ad.Call(func(_ []float64) {
			dist.Logps(0, 0, y...)
		}, 2, &alpha, &beta))
	}
}

func (beta) Logp(alpha, beta float64, y float64) float64 {
	if ad.Called() {
		ad.Enter(&alpha, &beta, &y)
	} else {
		panic("Logp called outside Observe")
	}
	return ad.Return(ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpSub, ad.Arithmetic(ad.OpSub, ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpMul, (ad.Arithmetic(ad.OpSub, &alpha, ad.Value(1))), ad.Elemental(math.Log, &y)), ad.Arithmetic(ad.OpMul, (ad.Arithmetic(ad.OpSub, &beta, ad.Value(1))), ad.Elemental(math.Log, ad.Arithmetic(ad.OpSub, ad.Value(1), &y)))), ad.Elemental(mathx.LogGamma, &alpha)), ad.Elemental(mathx.LogGamma, &beta)), ad.Elemental(mathx.LogGamma, ad.Arithmetic(ad.OpAdd, &alpha, &beta))))
}

func (beta) Logps(alpha, beta float64, y ...float64) float64 {
	if ad.Called() {
		ad.Enter(&alpha, &beta)
	} else {
		panic("Logps called outside Observe")
	}
	var lp float64
	ad.Assignment(&lp, ad.Arithmetic(ad.OpMul, (ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpSub, ad.Arithmetic(ad.OpNeg, ad.Elemental(mathx.LogGamma, &alpha)), ad.Elemental(mathx.LogGamma, &beta)), ad.Elemental(mathx.LogGamma, ad.Arithmetic(ad.OpAdd, &alpha, &beta)))), ad.Value(float64(len(y)))))
	for i := range y {
		ad.Assignment(&lp, ad.Arithmetic(ad.OpAdd, &lp, ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpMul, (ad.Arithmetic(ad.OpSub, &alpha, ad.Value(1))), ad.Elemental(math.Log, &y[i])), ad.Arithmetic(ad.OpMul, (ad.Arithmetic(ad.OpSub, &beta, ad.Value(1))), ad.Elemental(math.Log, ad.Arithmetic(ad.OpSub, ad.Value(1), &y[i]))))))
	}
	return ad.Return(&lp)
}

type binomial struct{}

var Binomial binomial

func (dist binomial) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	var (
		n	float64

		p	float64

		y	[]float64
	)

	n, p, y = x[0], x[1], x[2:]
	if len(y) == 1 {
		return ad.Return(ad.Call(func(_ []float64) {
			dist.Logp(0, 0, 0)
		}, 3, &n, &p, &y[0]))
	} else {
		return ad.Return(ad.Call(func(_ []float64) {
			dist.Logps(0, 0, y...)
		}, 2, &n, &p))
	}
}

func (binomial) Logp(n, p float64, y float64) float64 {
	if ad.Called() {
		ad.Enter(&n, &p, &y)
	} else {
		panic("Logp called outside Observe")
	}
	return ad.Return(ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpSub, ad.Arithmetic(ad.OpSub, ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpMul, &y, ad.Elemental(math.Log, &p)), ad.Arithmetic(ad.OpMul, (ad.Arithmetic(ad.OpSub, &n, &y)), ad.Elemental(math.Log, ad.Arithmetic(ad.OpSub, ad.Value(1), &p)))), ad.Elemental(mathx.LogGamma, ad.Arithmetic(ad.OpAdd, &y, ad.Value(1)))), ad.Elemental(mathx.LogGamma, ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpSub, &n, &y), ad.Value(1)))), ad.Elemental(mathx.LogGamma, ad.Arithmetic(ad.OpAdd, &n, ad.Value(1)))))
}

func (binomial) Logps(n, p float64, y ...float64) float64 {
	if ad.Called() {
		ad.Enter(&n, &p)
	} else {
		panic("Logps called outside Observe")
	}
	var lp float64
	ad.Assignment(&lp, ad.Arithmetic(ad.OpMul, ad.Elemental(mathx.LogGamma, ad.Arithmetic(ad.OpAdd, &n, ad.Value(1))), ad.Value(float64(len(y)))))
	for i := range y {
		ad.Assignment(&lp, ad.Arithmetic(ad.OpAdd, &lp, ad.Arithmetic(ad.OpSub, ad.Arithmetic(ad.OpSub, ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpMul, &y[i], ad.Elemental(math.Log, &p)), ad.Arithmetic(ad.OpMul, (ad.Arithmetic(ad.OpSub, &n, &y[i])), ad.Elemental(math.Log, ad.Arithmetic(ad.OpSub, ad.Value(1), &p)))), ad.Elemental(mathx.LogGamma, ad.Arithmetic(ad.OpAdd, &y[i], ad.Value(1)))), ad.Elemental(mathx.LogGamma, ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpSub, &n, &y[i]), ad.Value(1))))))
	}
	return ad.Return(&lp)
}

type Dirichlet struct {
	N int
}

var Dir Dirichlet

func (dist Dirichlet) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	var alpha []float64

	alpha = x[:dist.N]
	if len(x[dist.N:]) == dist.N {
		return ad.Return(ad.Call(func(_ []float64) {
			dist.Logp(alpha, x[dist.N:])
		}, 0))
	} else {
		var ys [][]float64

		ys = make([][]float64, len(x[dist.N:])/dist.N)
		for i := range ys {
			ys[i] = x[dist.N*(i+1) : dist.N*(i+2)]
		}
		return ad.Return(ad.Call(func(_ []float64) {
			dist.Logps(alpha, ys...)
		}, 0))
	}
}

func (dist Dirichlet) Logp(alpha []float64, y []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		panic("Logp called outside Observe")
	}
	var sum float64
	ad.Assignment(&sum, ad.Value(0.))
	for j := range y {
		ad.Assignment(&sum, ad.Arithmetic(ad.OpAdd, &sum, ad.Arithmetic(ad.OpMul, (ad.Arithmetic(ad.OpSub, &alpha[j], ad.Value(1))), ad.Elemental(math.Log, &y[j]))))
	}

	return ad.Return(ad.Arithmetic(ad.OpSub, &sum, ad.Call(func(_ []float64) {
		dist.logZ(alpha)
	}, 0)))
}

func (dist Dirichlet) Logps(alpha []float64, y ...[]float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		panic("Logps called outside Observe")
	}
	var logZ float64
	ad.Assignment(&logZ, ad.Call(func(_ []float64) {
		dist.logZ(alpha)
	}, 0))
	var lp float64
	ad.Assignment(&lp, ad.Arithmetic(ad.OpMul, ad.Arithmetic(ad.OpNeg, &logZ), ad.Value(float64(len(y)))))
	for i := range y {
		for j := range alpha {
			ad.Assignment(&lp, ad.Arithmetic(ad.OpAdd, &lp, ad.Arithmetic(ad.OpMul, (ad.Arithmetic(ad.OpSub, &alpha[j], ad.Value(1))), ad.Elemental(math.Log, &y[i][j]))))
		}
	}
	return ad.Return(&lp)
}

func (dist Dirichlet) logZ(alpha []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		panic("logZ called outside Observe")
	}
	var sumAlpha float64
	ad.Assignment(&sumAlpha, ad.Value(0.))
	var sumLogGammaAlpha float64
	ad.Assignment(&sumLogGammaAlpha, ad.Value(0.))
	for i := range alpha {
		ad.Assignment(&sumAlpha, ad.Arithmetic(ad.OpAdd, &sumAlpha, &alpha[i]))
		ad.Assignment(&sumLogGammaAlpha, ad.Arithmetic(ad.OpAdd, &sumLogGammaAlpha, ad.Elemental(mathx.LogGamma, &alpha[i])))
	}

	return ad.Return(ad.Arithmetic(ad.OpSub, &sumLogGammaAlpha, ad.Elemental(mathx.LogGamma, &sumAlpha)))
}

type bernoulli struct{}

var Bernoulli bernoulli

func (dist bernoulli) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	var (
		p	float64

		y	[]float64
	)

	p, y = x[0], x[1:]
	if len(y) == 1 {
		return ad.Return(ad.Call(func(_ []float64) {
			dist.Logp(0, 0)
		}, 2, &p, &y[0]))
	} else {
		return ad.Return(ad.Call(func(_ []float64) {
			dist.Logps(0, y...)
		}, 1, &p))
	}
}

func (bernoulli) Logp(p float64, y float64) float64 {
	if ad.Called() {
		ad.Enter(&p, &y)
	} else {
		panic("Logp called outside Observe")
	}
	if y >= 0.5 {
		return ad.Return(ad.Elemental(math.Log, &p))
	} else {
		return ad.Return(ad.Elemental(math.Log, ad.Arithmetic(ad.OpSub, ad.Value(1), &p)))
	}
}

func (bernoulli) Logps(p float64, y ...float64) float64 {
	if ad.Called() {
		ad.Enter(&p)
	} else {
		panic("Logps called outside Observe")
	}
	var lp float64
	ad.Assignment(&lp, ad.Value(0.))
	for i := range y {
		if y[i] >= 0.5 {
			ad.Assignment(&lp, ad.Arithmetic(ad.OpAdd, &lp, ad.Elemental(math.Log, &p)))
		} else {
			ad.Assignment(&lp, ad.Arithmetic(ad.OpAdd, &lp, ad.Elemental(math.Log, ad.Arithmetic(ad.OpSub, ad.Value(1), &p))))
		}
	}
	return ad.Return(&lp)
}

type Categorical struct {
	N int
}

var Cat Categorical

func (dist Categorical) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	if len(x) == dist.N+1 {
		return ad.Return(ad.Call(func(_ []float64) {
			dist.Logp(x[:dist.N], 0)
		}, 1, &x[dist.N]))
	} else {
		return ad.Return(ad.Call(func(_ []float64) {
			dist.Logps(x[:dist.N], x[dist.N:]...)
		}, 0))
	}
}

func (dist Categorical) Logp(
	alpha []float64, y float64,
) float64 {
	if ad.Called() {
		ad.Enter(&y)
	} else {
		panic("Logp called outside Observe")
	}
	var i int

	i = int(y)
	return ad.Return(ad.Arithmetic(ad.OpSub, ad.Elemental(math.Log, &alpha[i]), ad.Call(func(_ []float64) {
		dist.logZ(alpha)
	}, 0)))
}

func (dist Categorical) Logps(
	alpha []float64, y ...float64,
) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		panic("Logps called outside Observe")
	}
	var logZ float64
	ad.Assignment(&logZ, ad.Call(func(_ []float64) {
		dist.logZ(alpha)
	}, 0))
	var lp float64
	ad.Assignment(&lp, ad.Arithmetic(ad.OpMul, ad.Arithmetic(ad.OpNeg, &logZ), ad.Value(float64(len(y)))))
	for i := range y {
		ad.Assignment(&lp, ad.Arithmetic(ad.OpAdd, &lp, ad.Elemental(math.Log, &alpha[int(y[i])])))
	}
	return ad.Return(&lp)
}

func (dist Categorical) logZ(alpha []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		panic("logZ called outside Observe")
	}
	var z float64
	ad.Assignment(&z, ad.Value(0.))
	for _, a := range alpha {
		ad.Assignment(&z, ad.Arithmetic(ad.OpAdd, &z, &a))
	}
	return ad.Return(ad.Elemental(math.Log, &z))
}

type d struct{}

func (d) Observe(_ []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup([]float64{})
	}
	panic("should never be called")
}

var D d

func (d) SoftMax(x, p []float64) {
	if ad.Called() {
		ad.Enter()
	} else {
		panic("SoftMax called outside Observe")
	}
	if len(x) != len(p) {
		panic(fmt.Sprintf("lengths of x and p are different: "+
			"got len(x)=%v, len(p)=%v", len(x), len(p)))
	}
	var max float64
	ad.Assignment(&max, ad.Value(math.Inf(-1)))
	for i := range x {
		if x[i] > max {
			ad.Assignment(&max, &x[i])
		}
	}
	var z float64
	ad.Assignment(&z, ad.Value(0.))
	for i := range x {
		var q float64
		ad.Assignment(&q, ad.Elemental(math.Exp, ad.Arithmetic(ad.OpSub, &x[i], &max)))
		ad.Assignment(&z, ad.Arithmetic(ad.OpAdd, &z, &q))
		ad.Assignment(&p[i], &q)
	}
	for i := range p {
		ad.Assignment(&p[i], ad.Arithmetic(ad.OpDiv, &p[i], &z))
	}
}

func (d) LogSumExp(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		panic("LogSumExp called outside Observe")
	}
	var max float64
	ad.Assignment(&max, ad.Value(math.Inf(-1)))
	for i := range x {
		if x[i] > max {
			ad.Assignment(&max, &x[i])
		}
	}
	var sumExp float64
	ad.Assignment(&sumExp, ad.Value(0.))
	for i := range x {
		ad.Assignment(&sumExp, ad.Arithmetic(ad.OpAdd, &sumExp, ad.Elemental(math.Exp, ad.Arithmetic(ad.OpSub, &x[i], &max))))
	}

	return ad.Return(ad.Arithmetic(ad.OpAdd, &max, ad.Elemental(math.Log, &sumExp)))
}

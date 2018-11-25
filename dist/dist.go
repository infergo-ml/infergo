// Package dist provides differentiatable distribution models.
// The package is automatically differentiated by deriv during
// build.
package dist

import (
	"bitbucket.org/dtolpin/infergo/mathx"
	"fmt"
	"math"
)

// Normal distribution.
type normal struct{}

// Normal distribution, singleton instance.
var Normal normal

// Observe implements the Model interface. The parameter
// vector is mu, sigma, observations.
func (dist normal) Observe(x []float64) float64 {
	mu, sigma, y := x[0], x[1], x[2:]
	if len(y) == 1 {
		return dist.Logp(mu, sigma, y[0])
	} else {
		return dist.Logps(mu, sigma, y...)
	}
}

var log2pi float64

func init() {
	log2pi = math.Log(2. * math.Pi)
}

// Logp computes the log pdf of a single observation.
func (_ normal) Logp(mu, sigma float64, y float64) float64 {
	vari := sigma * sigma
	logv := math.Log(vari)
	d := y - mu
	return -0.5 * (d*d/vari + logv + log2pi)
}

// Logps computes the log pdf of a vector of observations.
func (_ normal) Logps(mu, sigma float64, y ...float64) float64 {
	vari := sigma * sigma
	logv := math.Log(vari)
	ll := 0.
	for i := range y {
		d := y[i] - mu
		ll += -0.5 * (d*d/vari + logv + log2pi)
	}
	return ll
}

// Exponential distribution.
type expon struct{}

// Exponential distribution, singleton instance.
var Expon expon

// Observe implements the Model interface. The parameter
// vector is lambda, observations.
func (dist expon) Observe(x []float64) float64 {
	lambda, y := x[0], x[1:]
	if len(y) == 1 {
		return dist.Logp(lambda, y[0])
	} else {
		return dist.Logps(lambda, y...)
	}
}

// Logp computes the log pdf of a single observation.
func (_ expon) Logp(lambda float64, y float64) float64 {
	logl := math.Log(lambda)
	return logl - lambda*y
}

// Logps computes the log pdf of a vector of observations.
func (_ expon) Logps(lambda float64, y ...float64) float64 {
	ll := 0.
	logl := math.Log(lambda)
	for i := range y {
		ll += logl - lambda*y[i]
	}
	return ll
}

// Gamma distribution
type gamma struct{}

// Gamma distribution, singleton instance.
var Gamma gamma

// Observe implements the Model interface. The parameter
// vector is alpha, beta, observations.
func (dist gamma) Observe(x []float64) float64 {
	alpha, beta, y := x[0], x[1], x[2:]
	if len(y) == 1 {
		return dist.Logp(alpha, beta, y[0])
	} else {
		return dist.Logps(alpha, beta, y...)
	}
}

// Logp computes the log pdf of a single observation.
func (_ gamma) Logp(alpha, beta float64, y float64) float64 {
	return (alpha-1)*math.Log(y) - beta*y -
		mathx.LogGamma(alpha) + alpha*math.Log(beta)
}

// Logps computes the log pdf of a vector of observations.
func (_ gamma) Logps(alpha, beta float64, y ...float64) float64 {
	ll := 0.
	for i := range y {
		ll += (alpha-1)*math.Log(y[i]) - beta*y[i] -
			mathx.LogGamma(alpha) + alpha*math.Log(beta)
	}
	return ll
}

// Beta distribution
type beta struct{}

// Beta distribution, singleton instance.
var Beta beta

// Observe implements the Model interface. The parameter
// vector is alpha, beta, observations.
func (dist beta) Observe(x []float64) float64 {
	alpha, beta, y := x[0], x[1], x[2:]
	if len(y) == 1 {
		return dist.Logp(alpha, beta, y[0])
	} else {
		return dist.Logps(alpha, beta, y...)
	}
}

// Logp computes the log pdf of a single observation.
func (_ beta) Logp(alpha, beta float64, y float64) float64 {
	return (alpha-1)*math.Log(y) +
		(beta-1)*math.Log(1-y) -
		mathx.LogGamma(alpha) - mathx.LogGamma(beta) +
		mathx.LogGamma(alpha+beta)
}

// Logp computes the log pdf of a vector of observations.
func (_ beta) Logps(alpha, beta float64, y ...float64) float64 {
	ll := 0.
	for i := range y {
		ll += (alpha-1)*math.Log(y[i]) +
			(beta-1)*math.Log(1-y[i]) -
			mathx.LogGamma(alpha) - mathx.LogGamma(beta) +
			mathx.LogGamma(alpha+beta)
	}
	return ll
}

// Dirichlet distribution
type Dirichlet struct {
	N int // number of dimensions
}

// Observe implements the Model interface. The parameters are
// alpha and observations,  flattened.
func (dist Dirichlet) Observe(x []float64) float64 {
	alpha := x[:dist.N]
	if len(x[dist.N:]) == dist.N {
		return dist.Logp(alpha, x[dist.N:])
	} else {
		ys := make([][]float64, len(x[dist.N:])/dist.N)
		for i := range ys {
			ys[i] = x[dist.N*(i+1) : dist.N*(i+2)]
		}
		return dist.Logps(alpha, ys...)
	}
}

// Logp computes logpdf of a single observation.
func (dist Dirichlet) Logp(alpha []float64, y []float64) float64 {
	sum := 0.
	for j := range y {
		sum += (alpha[j] - 1.) * math.Log(y[j])
	}

	return dist.logZ(alpha) + sum
}

// Logps computes logpdf of a vector of observations.
func (dist Dirichlet) Logps(alpha []float64, y ...[]float64) float64 {
	ll := 0.
	logZ := dist.logZ(alpha)
	for i := range y {
		ll += logZ
		sum := 0.
		for j := range alpha {
			sum += (alpha[j] - 1.) * math.Log(y[i][j])
		}
		ll += sum
	}
	return ll
}

// SoftMax transforms unconstrained parameters to a point on the
// unit hyperplane suitable to be observed from Dirichlet. x is
// the original vector, p is a point on the unit hyperplane.
func (dist Dirichlet) SoftMax(x, p []float64) {
	if len(x) != len(p) {
		panic(fmt.Sprintf("lengths of x and p are different: "+
			"got len(x)=%v, len(p)=%v", len(x), len(p)))
	}

	// For a more stable computation, first find max(x) and
	// then divide both numerator and denominator of SoftMax
	// by exp(xmax).
	xmax := math.Inf(-1)
	for i := range x {
		if x[i] > xmax {
			xmax = x[i]
		}
	}

	// Transform and normalize components.
	z := 0.
	for i := range x {
		q := math.Exp(x[i] - xmax)
		z += q
		p[i] = q
	}
	for i := range p {
		p[i] /= z
	}
}

// Outside of differentiated context, SoftMax can be used
// without distribution.
var SoftMax func(x, p []float64)

func init() {
	SoftMax = Dirichlet{}.SoftMax
}

// logZ computes normalization term independent of observations.
func (dist Dirichlet) logZ(alpha []float64) float64 {
	sumAlpha := 0.
	sumLogGammaAlpha := 0.
	for i := range alpha {
		sumAlpha += alpha[i]
		sumLogGammaAlpha += mathx.LogGamma(alpha[i])
	}

	return mathx.LogGamma(sumAlpha) - sumLogGammaAlpha
}

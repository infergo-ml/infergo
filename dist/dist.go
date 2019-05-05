// Package dist provides differentiatable distribution models.
// The package is automatically differentiated by deriv during
// build.
package dist

import (
	"bitbucket.org/dtolpin/infergo/mathx"
	"fmt"
	"math"
)

// Common constants

var (
	logpi, log2pi float64
)

func init() {
	log2 := math.Log(2)
	logpi = math.Log(math.Pi)
	log2pi = log2 + logpi
}


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
	ll := -0.5 * (logv + log2pi) * float64(len(y))
	for i := range y {
		d := y[i] - mu
		ll -= 0.5 * d*d/vari
	}
	return ll
}

// Cauchy distribution
type cauchy struct{}

// Cauchy distribution, singleton instance.
var Cauchy cauchy

// Observe implements the Model interface. The parameter
// vector is mu, sigma, observations.
func (dist cauchy) Observe(x []float64) float64 {
	mu, sigma, y := x[0], x[1], x[2:]
	if len(y) == 1 {
		return dist.Logp(mu, sigma, y[0])
	} else {
		return dist.Logps(mu, sigma, y...)
	}
}

// Logp computes the log pdf of a single observation.
func (_ cauchy) Logp(x0, gamma float64, y float64) float64 {
	logGamma := math.Log(gamma)
	d := (y - x0) / gamma
	return - logGamma - logpi - math.Log(1 + d*d)
}

// Logps computes the log pdf of a vector of observations.
func (_ cauchy) Logps(x0, gamma float64, y ...float64) float64 {
	logGamma := math.Log(gamma)
	ll := (- logGamma - logpi)*float64(len(y))
	for i := range y {
		d := (y[i] - x0) / gamma
		ll -= math.Log(1 + d*d)
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
	logl := math.Log(lambda)
	ll := logl*float64(len(y))
	for i := range y {
		ll -= lambda*y[i]
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
	ll := (- mathx.LogGamma(alpha) +
	  	   alpha*math.Log(beta))*float64(len(y))
	for i := range y {
		ll += (alpha-1)*math.Log(y[i]) - beta*y[i]
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
	ll := (- mathx.LogGamma(alpha) - mathx.LogGamma(beta) +
     	   mathx.LogGamma(alpha+beta)) * float64(len(y))
	for i := range y {
		ll += (alpha-1)*math.Log(y[i]) +
			(beta-1)*math.Log(1-y[i])
	}
	return ll
}

// Dirichlet distribution
type Dirichlet struct {
	N int // number of dimensions
}

// Observe implements the Model interface. The parameters are
// alpha and observations, flattened.
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
		sum += (alpha[j] - 1) * math.Log(y[j])
	}

	return sum - dist.logZ(alpha)
}

// Logps computes logpdf of a vector of observations.
func (dist Dirichlet) Logps(alpha []float64, y ...[]float64) float64 {
	logZ := dist.logZ(alpha)
	ll := -logZ * float64(len(y))
	for i := range y {
		for j := range alpha {
			ll += (alpha[j] - 1) * math.Log(y[i][j])
		}
	}
	return ll
}

// logZ computes the normalization constant.
func (dist Dirichlet) logZ(alpha []float64) float64 {
	sumAlpha := 0.
	sumLogGammaAlpha := 0.
	for i := range alpha {
		sumAlpha += alpha[i]
		sumLogGammaAlpha += mathx.LogGamma(alpha[i])
	}

	return sumLogGammaAlpha - mathx.LogGamma(sumAlpha)
}

// the categorical distribution
type Categorical struct {
	N int // number of categories
}

// Observe implements the Model interface
func (dist Categorical) Observe(x []float64) float64 {
	if len(x) == dist.N+1 {
		return dist.Logp(x[:dist.N], x[dist.N])
	} else {
		return dist.Logps(x[:dist.N], x[dist.N:]...)
	}
}

// Logp computes logpdf of a single observation.
func (dist Categorical) Logp(
	alpha []float64, y float64,
) float64 {
	i := int(y)
	return math.Log(alpha[i]) - dist.logZ(alpha)
}

// Logps computes logpdf of a vector of observations.
func (dist Categorical) Logps(
	alpha []float64, y ...float64,
) float64 {
	logZ := dist.logZ(alpha)
	ll := -logZ * float64(len(y))
	for i := range y {
		ll += math.Log(alpha[int(y[i])])
	}
	return ll
}

// logZ computes the normalization constant.
func (dist Categorical) logZ(alpha []float64) float64 {
	z := 0.
	for _, a := range alpha {
		z += a
	}
	return math.Log(z)
}

// Differentiable functions not belonging to a distribution

// Type d is a placeholder for differentiated functions without
// a model.
type d struct{}

// Method Observe implements the Model interface on d and
// makes d's methods differentiable.
func (_ d) Observe(_ []float64) float64 {
	panic("should never be called")
}

// D is a singletone variable of type d. General log-likelihood
// handling functions are dispatched on d.
var D d

// SoftMax transforms unconstrained parameters x to a point p on
// the simplex.
func (_ d) SoftMax(x, p []float64) {
	if len(x) != len(p) {
		panic(fmt.Sprintf("lengths of x and p are different: "+
			"got len(x)=%v, len(p)=%v", len(x), len(p)))
	}

	// For a more stable computation, first find max(x) and
	// then divide both numerator and denominator of SoftMax
	// by exp(max).
	max := math.Inf(-1)
	for i := range x {
		if x[i] > max {
			max = x[i]
		}
	}

	// Transform and normalize components.
	z := 0.
	for i := range x {
		q := math.Exp(x[i] - max)
		z += q
		p[i] = q
	}
	for i := range p {
		p[i] /= z
	}
}

// LogSumExp computes log(sum(exp(x[0]) + exp(x[1]) + ...) robustly.
func (_ d) LogSumExp(x []float64) float64 {
	max := math.Inf(-1)
	for i := range x {
		if x[i] > max {
			max = x[i]
		}
	}

	sumExp := 0.
	for i := range x {
		sumExp += math.Exp(x[i] - max)
	}

	return max + math.Log(sumExp)
}

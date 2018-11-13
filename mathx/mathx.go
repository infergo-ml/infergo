package mathx

import (
	"bitbucket.org/dtolpin/infergo/ad"
	"math"
)

// Sigmoid computes the sigmoid function 1/(1 + exp(-x)).
func Sigm(x float64) float64 {
	return 1. / (1. + math.Exp(-x))
}

func init() {
	ad.RegisterElemental(Sigm,
		// dSigm / dx = Exp(-x) / (1 + Exp(-x))^2
		//            = Sigm(x) * (1 - Sigm(x))
		func(value float64, _ ...float64) []float64 {
			return []float64{value * (1. - value)}
		})
}

// LogSumExp computes log(exp(x) + exp(y)) robustly.
func LogSumExp(x, y float64) float64 {
	z := x
	if y > z {
		z = y
	}
	return z + math.Log(math.Exp(x-z)+math.Exp(y-z))
}

func init() {
	// d lse(x, y) / dx = exp(x) / exp(x) + exp(y)
	//                  = 1 / 1 + exp(y - x)
	// d lse(x, y) / dy = exp(y) / exp(x) + exp(y)
	//                  = exp(y - x) / 1 + exp(y - x)
	ad.RegisterElemental(LogSumExp,
		func(_ float64, params ...float64) []float64 {
			z := math.Exp(params[1] - params[0])
			t := 1. / (1. + z)
			return []float64{t, t * z}
		})
}

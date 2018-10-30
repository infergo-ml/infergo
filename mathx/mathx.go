package mathx

import (
	"math"
)

// Sigmoid computes the sigmoid function 1/(1 + exp(-x)).
func Sigm(x float64) float64 {
	return 1. / (1. + math.Exp(-x))
}

// LogSumExp computes log(exp(x) + exp(y)) robustly.
func LogSumExp(x, y float64) float64 {
	z := x
	if y > z {
		z = y
	}
	return z + math.Log(math.Exp(x - z) + math.Exp(y - z))
}

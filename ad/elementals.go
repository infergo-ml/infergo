package ad

import (
	"math"
	"reflect"
)

// The gradient of an elemental function accepts the function value
// and the parameters and returns a vector of partial gradients.
// Depending on the function, either the value or the parameters
// may be ignored in the computation of the gradient.
type gradient func(value float64, params ...float64) []float64

var elementals map[uintptr]gradient

// Function fkey computes map key for a function.
func fkey(f interface{}) uintptr {
	return reflect.ValueOf(f).Pointer()
}

// Function RegisterElemental registers the gradient
// for an elemental function.
func RegisterElemental(f interface{}, g gradient) {
	elementals[fkey(f)] = g
}

// Function Elementalgradient returns the gradient for a function.
// If the function is not registered as elemental, the second returned
// value is false.
func ElementalGradient(f interface{}) (gradient, bool) {
	g, ok := elementals[fkey(f)]
	return g, ok
}

// Elementals from the math package.
func init() {
	elementals = make(map[uintptr]gradient)
	RegisterElemental(math.Sqrt,
		func(value float64, _ ...float64) []float64 {
			return []float64{0.5 / value}
		})
	RegisterElemental(math.Exp,
		func(value float64, _ ...float64) []float64 {
			return []float64{value}
		})
	RegisterElemental(math.Log,
		func(_ float64, params ...float64) []float64 {
			return []float64{1. / params[0]}
		})
	RegisterElemental(math.Sin,
		func(_ float64, params ...float64) []float64 {
			return []float64{math.Cos(params[0])}
		})
	RegisterElemental(math.Cos,
		func(_ float64, params ...float64) []float64 {
			return []float64{-math.Sin(params[0])}
		})
}

// Additional elementals useful in learning programs.

// Sigmoid
func Sigm(x float64) float64 {
	return 1. / (1. + math.Exp(-x))
}

func init() {
	RegisterElemental(Sigm,
		// dSigm / dx = Exp(-x) / (1 + Exp(-x))^2
		//            = Sigm(x) * (1 - Sigm(x))
		func(value float64, _ ...float64) []float64 {
			return []float64{value * (1. - value)}
		})
}

// Logarithm of union probability
func LogSumExp(x, y float64) float64 {
	z := x
	if y > z {
		z = y
	}
	return z + math.Log(math.Exp(x - z) + math.Exp(y - z))
}

func init() {
	// d lse(x, y) / dx = exp(x) / exp(x) + exp(y) 
	//                  = 1 / 1 + exp(y - x)
	// d lse(x, y) / dy = exp(y) / exp(x) + exp(y)
	//                  = exp(y - x) / 1 + exp(y - x)
	RegisterElemental(LogSumExp,
		func(_ float64, params ...float64) []float64 {
			z := math.Exp(params[1] - params[0])
			t := 1. / (1. + z)
			return []float64{t, t * z}
		})
}

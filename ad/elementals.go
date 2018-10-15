package ad

import (
	"math"
	"reflect"
)

// The gradient of an elemental function accepts the function value
// and the parameters and returns a vector of partial gradients.
// Depending on the function, either the value or the parameters
// may be ignored in the computation of the gradient.
type gradient func(value float64, parameters ...float64) []float64

var elementals map[uintptr]gradient

// Function fkey computes map key for a function.
func fkey(f interface{}) uintptr {
	return reflect.ValueOf(f).Pointer()
}

// Function RegisterElemental registers the deriv
// for an elemental function.
func RegisterElemental(f interface{}, g gradient) {
	elementals[fkey(f)] = g
}

// Function Elementalgradient returns the deriv for a function.
// If the function is not registered as elemental, the second returned
// value is false.
func Elementalgradient(f interface{}) (gradient, bool) {
	deriv, ok := elementals[fkey(f)]
	return deriv, ok
}

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
		func(_ float64, parameters ...float64) []float64 {
			return []float64{1. / parameters[0]}
		})
	RegisterElemental(math.Sin,
		func(_ float64, parameters ...float64) []float64 {
			return []float64{math.Cos(parameters[0])}
		})
	RegisterElemental(math.Cos,
		func(_ float64, parameters ...float64) []float64 {
			return []float64{-math.Sin(parameters[0])}
		})
}

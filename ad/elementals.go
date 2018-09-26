package ad

import (
	"math"
	"reflect"
)

type Derivative func(value float64, parameters ...float64) float64
type Elementals map[uintptr]Derivative

var elementals Elementals

// Function fkey computes map key for a function.
func fkey(function interface{}) uintptr {
	return reflect.ValueOf(function).Pointer()
}

// Function RegisterElemental registers the derivative
// for an elemental function.
func RegisterElemental(function interface{}, derivative Derivative) {
	elementals[fkey(function)] = derivative
}

// Function ElementalDerivative returns the derivative for a function.
// If the function is not registered as elemental, the second returned
// value is false.
func ElementalDerivative(function interface{}) (Derivative, bool) {
	derivative, ok := elementals[fkey(function)]
	return derivative, ok
}

func init() {
	elementals = make(Elementals)
	RegisterElemental(math.Sqrt, func(value float64, parameters ...float64) float64 {
		return 0.5 / value
	})
	RegisterElemental(math.Exp, func(value float64, parameters ...float64) float64 {
		return value
	})
	RegisterElemental(math.Log, func(value float64, parameters ...float64) float64 {
		return 1. / parameters[0]
	})
	RegisterElemental(math.Sin, func(value float64, parameters ...float64) float64 {
		return math.Cos(parameters[0])
	})
	RegisterElemental(math.Cos, func(value float64, parameters ...float64) float64 {
		return -math.Sin(parameters[0])
	})
}

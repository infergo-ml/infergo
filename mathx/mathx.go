// Package mathx provides auxiliary elemental functions,
// ubiquitously useful but not found in package math.
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
			t := 1 / (1 + z)
			return []float64{t, t * z}
		})
}

// LogGamma and digamma are borrowed from the source code of
// WebPPL, https://github.com/probmods/webppl.
// Copyright © 2014 WebPPL contributors

// LogGamma is used in the log-density of the Gamma and Beta
// distributions.
func LogGamma(x float64) float64 {
	x -= 1
	tmp := x + 5.5
	tmp -= (x + 0.5) * math.Log(tmp)
	var ser = 1.000000000190015
	for _, g := range gammaCof {
		x += 1
		ser += g / x
	}
	return -tmp + math.Log(2.5066282746310005*ser)
}

var gammaCof = []float64{
	76.18009172947146,
	-86.50532032941677,
	24.01409824083091,
	-1.231739572450155,
	0.1208650973866179e-2,
	-0.5395239384953e-5,
}

// digamma is the derivative of LogGamma.
func digamma(x float64) float64 {
	if x < 6 {
		return digamma(x+1) - 1/x
	}
	return math.Log(x) -
		1/(2*x) -
		1/(12*math.Pow(x, 2)) +
		1/(120*math.Pow(x, 4)) -
		1/(252*math.Pow(x, 6)) +
		1/(240*math.Pow(x, 8)) -
		5/(660*math.Pow(x, 10)) +
		691/(32760*math.Pow(x, 12)) -
		1/(12*math.Pow(x, 14))
}

func init() {
	ad.RegisterElemental(LogGamma,
		func(_ float64, params ...float64) []float64 {
			return []float64{digamma(params[0])}
		})
}

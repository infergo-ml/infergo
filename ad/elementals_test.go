package ad

// Testing the elementals

import (
	"bitbucket.org/dtolpin/infergo/mathx"
	"math"
	"testing"
)

// Check that gradients on some functions from math
// are defined and correct.
func TestMath(t *testing.T) {
	for _, c := range []struct {
		s  string
		f  func(float64) float64
		xg [][2]float64
	}{
		{
			"sqrt",
			math.Sqrt,
			[][2]float64{{0.25, 1.}, {4., 0.25}},
		},
		{
			"exp",
			math.Exp,
			[][2]float64{{0., 1.}, {2., math.Exp(2.)}},
		},
		{
			"log",
			math.Log,
			[][2]float64{{0.5, 2.}, {2., 0.5}},
		},
		{
			"sin",
			math.Sin,
			[][2]float64{{0., 1.}, {math.Pi, -1.}},
		},
		{
			"cos",
			math.Cos,
			[][2]float64{{0., 0.}, {math.Pi, 0.}},
		},
	} {
		grad, ok := ElementalGradient(c.f)
		if !ok {
			t.Errorf("No gradient for %v", c.s)
		}
		for _, xg := range c.xg {
			y := c.f(xg[0])
			g := grad(y, xg[0])[0]
			if math.Abs(g-xg[1]) > 1.0E-6 {
				t.Errorf("Wrong gradient of %v: "+
					"got %v(%.4g)=%.4g, want %.4g",
					c.s, c.s, xg[0], g, xg[1])
			}
		}
	}
}

// Check the gradient of Sigm.
func TestSigm(t *testing.T) {
	grad, ok := ElementalGradient(mathx.Sigm)
	if !ok {
		t.Errorf("No gradient for Sigm")
	}
	for _, c := range []struct {
		x, g float64
	}{
		{0., 0.25},
		{20., 0.},
		{-20., 0.},
	} {
		y := mathx.Sigm(c.x)
		g := grad(y, c.x)[0]
		if math.Abs(g-c.g) > 1.0E-6 {
			t.Errorf("Wrong gradient of Sigm: "+
				"got Sigm(%.4g)=%.4g, want %.4g",
				c.x, g, c.g)
		}
	}
}

// Check the gradient of LogSumExp.
func TestLogSumExp(t *testing.T) {
	grad, ok := ElementalGradient(mathx.LogSumExp)
	if !ok {
		t.Errorf("No gradient for LogSumExp")
	}
	for _, c := range []struct {
		x, y float64
		g    [2]float64
	}{
		{0., 0., [2]float64{0.5, 0.5}},
		{1., 1., [2]float64{0.5, 0.5}},
		{0., 0.5, [2]float64{0.3775407, 0.6224593}},
		{0.5, 0., [2]float64{0.6224593, 0.3775407}},
	} {
		z := mathx.LogSumExp(c.x, c.x)
		g := grad(z, c.x, c.y)
		if math.Abs(g[0]-c.g[0]) > 1.0E-6 ||
			math.Abs(g[1]-c.g[1]) > 1.0E-6 {
			t.Errorf("Wrong gradient of LogSumExp: "+
				"got LogSumExp(%.4g, %.4g)=(%.4g, %.4g), "+
				"want (%.4g, %.4g)",
				c.x, c.y, g[0], g[1], c.g[0], c.g[1])
		}
	}
}

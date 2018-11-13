package ad

// Testing the elementals

import (
	"math"
	"testing"
)

// Check that gradients of unary functions from math
// are defined and correct.
func TestMathUnary(t *testing.T) {
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
		{
			"tan",
			math.Tan,
			[][2]float64{{0., 1.}, {0.25 * math.Pi, 2.}},
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
				t.Errorf("Wrong gradient of %v(%.4g): "+
					"got %.4g, want %.4g", c.s, xg[0], g, xg[1])
			}
		}
	}
}

// Check that gradients of binary functions from math
// are defined and correct.
func TestMathBinary(t *testing.T) {
	for _, c := range []struct {
		s  string
		f  func(float64, float64) float64
		xg [][3]float64
	}{
		{
			"pow",
			math.Pow,
			[][3]float64{
				{1., 0., 0.},
				{math.E, 0., 1.},
				{2., 2., 4. * math.Log(2.)},
			},
		},
	} {
		grad, ok := ElementalGradient(c.f)
		if !ok {
			t.Errorf("No gradient for %v", c.s)
		}
		for _, xg := range c.xg {
			y := c.f(xg[0], xg[1])
			g := grad(y, xg[0], xg[1])[0]
			if math.Abs(g-xg[2]) > 1.0E-6 {
				t.Errorf("Wrong gradient of %v(%.4g, %.4g): "+
					"got %.4g, want %.4g",
					c.s, xg[0], xg[1], g, xg[2])
			}
		}
	}
}

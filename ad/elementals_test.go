package ad

// Testing the elementals

import (
	"math"
	"testing"
)

func TestMath(t *testing.T) {
	for _, c := range []struct {
		s string
		f func(float64) float64
		xg [][2]float64
	} {
		{
			"sqrt",
			math.Sqrt,
			[][2]float64 {{0.25, 1.}, {4., 0.25}},
		},
		{
			"exp",
			math.Exp,
			[][2]float64 {{0., 1.}, {2., math.Exp(2.)}},
		},
		{
			"log",
			math.Log,
			[][2]float64 {{0.5, 2.}, {2., 0.5}},
		},
		{
			"sin",
			math.Sin,
			[][2]float64 {{0., 1.}, {math.Pi, -1.}},
		},
		{
			"cos",
			math.Cos,
			[][2]float64 {{0., 0.}, {math.Pi, 0.}},
		},
	} {
		grad, ok := ElementalGradient(c.f)
		if !ok {
			t.Errorf("No gradient for %v", c.s)
		}
		for _, xg := range c.xg {
			y := c.f(xg[0])
			g := grad(y, xg[0])[0]
			if math.Abs(g - xg[1]) > 1.0E-6 {
				t.Errorf("Wrong gradient of %v:" + 
					" got %v(%.4g)=%.4g, want %.4g",
					c.s, c.s, xg[0], g, xg[1])
			}
		}
	}
}


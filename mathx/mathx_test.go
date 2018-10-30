package mathx

import (
	"testing"
	"math"
)

func TestSigm(t *testing.T) {
	for _, c := range []struct {
		x, y float64
	} {
		{0., 0.5},
		{-100., 0.},
		{100., 1.},
	} {
		y := Sigm(c.x)
		if math.Abs(y - c.y) > 1E-6 {
			t.Errorf("Wrong Sigm(%.4g): got %.4g, want %.4g",
				c.x, y, c.y)
		}
	}
}

func TestLogSumExp(t *testing.T) {
	for _, c := range []struct {
		x, y, z  float64
	} {
		{ 0.,  0.,  0.693147181},
		{-1., -1., -0.306852819},
		{ 0.,  1.,  1.313261687}, 
		{ 1.,  0.,  1.313261687},
	} {
		z := LogSumExp(c.x, c.y)
		if math.Abs(z - c.z) > 1E-6 {
			t.Errorf("Wrong LogSumExp(%.4g, %.4g): got %.4g, want %.4g",
				c.x, c.y, z, c.z)
		}
	}
}

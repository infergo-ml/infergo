package mathx

import (
	"bitbucket.org/dtolpin/infergo/ad"
	"math"
	"testing"
)

func TestSigm(t *testing.T) {
	for _, c := range []struct {
		x, y float64
	}{
		{0., 0.5},
		{-100., 0.},
		{100., 1.},
	} {
		y := Sigm(c.x)
		if math.Abs(y-c.y) > 1E-6 {
			t.Errorf("Wrong Sigm(%.4g): got %.4g, want %.4g",
				c.x, y, c.y)
		}
	}
}

func TestSigmGrad(t *testing.T) {
	grad, ok := ad.ElementalGradient(Sigm)
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
		y := Sigm(c.x)
		g := grad(y, c.x)[0]
		if math.Abs(g-c.g) > 1.0E-6 {
			t.Errorf("Wrong gradient of Sigm: "+
				"got Sigm(%.4g)=%.4g, want %.4g",
				c.x, g, c.g)
		}
	}
}

func TestLogSumExp(t *testing.T) {
	for _, c := range []struct {
		x, y, z float64
	}{
		{0., 0., 0.693147181},
		{-1., -1., -0.306852819},
		{0., 1., 1.313261687},
		{1., 0., 1.313261687},
	} {
		z := LogSumExp(c.x, c.y)
		if math.Abs(z-c.z) > 1E-6 {
			t.Errorf("Wrong LogSumExp(%.4g, %.4g): got %.4g, want %.4g",
				c.x, c.y, z, c.z)
		}
	}
}

func TestLogSumExpGrad(t *testing.T) {
	grad, ok := ad.ElementalGradient(LogSumExp)
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
		z := LogSumExp(c.x, c.x)
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

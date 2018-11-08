package infer

// Testing MCMC algorithms.

import (
	"bitbucket.org/dtolpin/infergo/ad"
	"math"
	"testing"
)

// Unit tests for helpers

func TestEnergy(t *testing.T) {
	for _, c := range []struct {
		l float64
		r []float64
		e float64
	}{
		{1., []float64{0.}, 1},
		{1., []float64{1., 3.}, 6},
	} {
		if e := energy(c.l, c.r); math.Abs(e-c.e) > 1E-6 {
			t.Errorf("incorrect energy for l=%v, r=%v: "+
				"got=%.6g, want=%.6g", c.l, c.r, e, c.e)
		}
	}
}

func TestLeapfrog(t *testing.T) {
	// constGrad is defined in infer/infer_test.go
	m := &constGrad{
		grad: []float64{0.5, 1.5},
	}
	x, r, eps := []float64{0., 0.}, []float64{1., -1.}, 0.5
	m.Observe(x)
	grad := ad.Gradient()
	leapfrog(m, &grad, x, r, eps)
	xNext, rNext := []float64{0.5625, -0.3125}, []float64{1.25, -0.25}
	for i := 0; i != len(x); i++ {
		if math.Abs(x[i]-xNext[i]) > 1E-6 {
			t.Errorf("wrong leapfrog step: got x[%d] = %.6g, "+
				"want %.6g", i, x[i], xNext[i])
		}
	}
	for i := 0; i != len(x); i++ {
		if math.Abs(r[i]-rNext[i]) > 1E-6 {
			t.Errorf("wrong leapfrog step: got r[%d] = %.6g, "+
				"want %.6g", i, r[i], rNext[i])
		}
	}
}

func TestUTurn(t *testing.T) {
	for _, c := range []struct {
		xl, rl, xr, rr []float64
		uturn          bool
	}{
		{
			[]float64{-1., 0}, []float64{0., 1.},
			[]float64{1., 0.}, []float64{1., 0.},
			false,
		},
		{
			[]float64{-1., 0}, []float64{0., 1.},
			[]float64{-1., 0.}, []float64{1., 0.},
			false,
		},
		{
			[]float64{1., 0}, []float64{0., 1.},
			[]float64{0., 1.}, []float64{1., 0.},
			true,
		},
		{
			[]float64{1., 0}, []float64{0., 1.},
			[]float64{0., 1.}, []float64{-1., 0.},
			false,
		},
	} {
		if c.uturn {
			if !uTurn(c.xl, c.rl, c.xr, c.rr) {
				t.Errorf("missed uturn: %+v", c)
			}
		} else {
			if uTurn(c.xl, c.rl, c.xr, c.rr) {
				t.Errorf("false uturn: %+v", c)
			}
		}
	}
}

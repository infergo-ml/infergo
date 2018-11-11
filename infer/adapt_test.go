package infer

// Testing adaptation.

import (
	"math"
	"testing"
)

func TestStep(t *testing.T) {
	for _, c := range []struct {
		Mu, Eta, Kappa, t, x, grad float64
		xnext                      float64
	}{
		{0, 1, 0, 1, 0, 1, -1},
		{1, 1, 0, 1, 1, 1, 0},
		{1, 0.1, 0, 1, 1, 1, 0.9},
		{1, 1, 0, 2, 0, 0, 0.5},
	} {
		da := &DualAveraging{
			Mu:    c.Mu,
			Eta:   c.Eta,
			Kappa: c.Kappa,
		}
		xnext := da.Step(c.t, c.x, c.grad)
		if math.Abs(xnext-c.xnext) > 1E-6 {
			t.Errorf("wrong adaptation: got %.4g, want %.4g",
				xnext, c.xnext)
		}
	}
}

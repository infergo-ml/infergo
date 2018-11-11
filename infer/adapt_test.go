package infer

// Testing adaptation.

import (
	"testing"
	"math"
)

func TestDualAveraging(t *testing.T) {
	for _, c := range []struct {
		t, x, gradSum, Rate float64
		xnext float64
	} {
		{1, 0, 1, 0.1, -0.1},
		{4, 1, -2, 0.4, 1.},
	} {
		da := DualAveraging{Rate: c.Rate}
		xnext := da.Step(c.t, c.x, c.gradSum)
		if math.Abs(xnext - c.xnext) > 1E-6 {
			t.Errorf("wrong update for %+v: got %.4g, want %.4g",
				c, xnext, c.xnext)
		}
	}
}

package infer

import (
	"math"
)

// Adaptation

type DualAveraging struct {
	Rate float64
}

// Nesterov's primal-dual, oversimplified.
// chi = -gradSum/math.Sqrt(t)
// eta = Rate/t
// x = eta*chi + (1-eta)*x
func (da *DualAveraging) Step(t, x, gradSum float64) float64 {
	chi := -gradSum/math.Sqrt(t)
	eta := da.Rate/t
	return eta*chi + (1-eta)*x
}

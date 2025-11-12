package infer

import (
	"log"
	"math"
)

// Adaptation

// Parameters of dual averaging.
type DualAveraging struct {
	Rate float64
}

// Step implements Nesterov's primal-dual averaging,
// oversimplified.
//
//	chi = -gradSum/math.Sqrt(t)
//	eta = Rate/t
//	x = eta*chi + (1-eta)*x
func (da *DualAveraging) Step(t, x, gradSum float64) float64 {
	chi := -gradSum / math.Sqrt(t)
	eta := da.Rate / t
	return eta*chi + (1-eta)*x
}

// Parameters of adaptation to the target depth
type DepthAdapter struct {
	DualAveraging
	Depth   float64
	NAdpt   int
	MinGrad float64
}

// Adapt adapts NUTS sampler to the target  depth.  At most
// nIter iterations are run.
func (da *DepthAdapter) Adapt(
	nuts *NUTS,
	samples <-chan []float64,
	nIter int,
) {
	da.setDefaults()
	gradSum := 0.
	for i := 0; i != nIter; i++ {
		if len(<-samples) == 0 {
			break
		}
		if (i+1)%da.NAdpt == 0 {
			t := float64(i / da.NAdpt)
			Eps := nuts.Eps
			depth := nuts.MeanDepth()
			if t == 0 {
				// Guess initial value.
				// Step is roughly inverse proportional to depth.
				nuts.Eps *= depth / da.Depth
			} else {
				grad := (da.Depth - depth) / da.Depth
				if math.Abs(grad) < da.MinGrad {
					break
				}
				gradSum += grad
				nuts.Eps = da.Step(t, nuts.Eps, gradSum)
			}
			log.Printf("Adapting: depth: %.4g, step: %.4g => %.4g",
				depth, Eps, nuts.Eps)
			if i+da.NAdpt < nIter {
				nuts.Depth = nil // forget the depth
			}
		}
	}
	log.Printf("Adapted: %.4g, step: %.4g",
		nuts.MeanDepth(), nuts.Eps)
}

// setDefaults sets defaults for DepthAdapter fields.
func (da *DepthAdapter) setDefaults() {
	if da.Depth == 0 {
		da.Depth = 5
	}
	if da.NAdpt == 0 {
		da.NAdpt = 10
	}
	if da.MinGrad == 0 {
		da.MinGrad = 0.01
	}
}

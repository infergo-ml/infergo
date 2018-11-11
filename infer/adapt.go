package infer

import (
	"math"
)

// Adaptation

// Gradient-based dual averaging:
//  chi[t+1] <-  Mu - sqrt(t) * grad
//  eta[t] <- Eta * t^(-1-Kappa)
//  x[t+1] <- eta[t] * chi[t + 1] + (1 - eta[t]) * x[t]
// Default values for Mu and Kappa are usable, but Eta must be
// specified.
type DualAveraging struct {
	Mu    float64 // Attractor
	Eta   float64 // Learning rate
	Kappa float64 // Adaptation decay
}

// Step accepts the current time and value, and the gradient,
// and returns the updated value.
func (da *DualAveraging) Step(t, x, grad float64) float64 {
	chi := da.Mu - math.Sqrt(t)*grad
	eta := da.Eta * math.Pow(t, -1.-da.Kappa)
	x = eta*chi + (1.-eta)*x
	return x
}

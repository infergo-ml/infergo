package infer

import (
	"bitbucket.org/dtolpin/infergo/ad"
	"bitbucket.org/dtolpin/infergo/model"
	"sync"
)

// FuncGrad returns the function to minimize and the gradient,
// suitable as fields for gonum optimize.Problem, corresponding
// to maximization of the model's log-likelihood.
func FuncGrad(m model.Model) (
	Func func(x []float64) float64,
	Grad func(grad, x []float64),
) {
	_, isElemental := m.(model.ElementalModel)
	if ad.IsMTSafe() && !isElemental {
		// It is safe to run multiple differentiations in
		// parallel, no locking.

		Func = func(x []float64) float64 {
			ll := m.Observe(x)
			model.DropGradient(m)
			return -ll
		}

		Grad = func(grad, x []float64) {
			_, grad_ := m.Observe(x), model.Gradient(m)
			for i := range grad_ {
				grad[i] = -grad_[i]
			}
		}
	} else {
		// Either the tape must be locked or the model
		// is elemental.

		Func = func(x []float64) float64 {
			tapeMutex.Lock()
			ll := m.Observe(x)
			model.DropGradient(m)
			tapeMutex.Unlock()
			return -ll
		}

		Grad = func(grad, x []float64) {
			tapeMutex.Lock()
			_, grad_ := m.Observe(x), model.Gradient(m)
			tapeMutex.Unlock()
			for i := range grad_ {
				grad[i] = -grad_[i]
			}
		}
	}

	return Func, Grad
}

// Tape mutex is used if the tape is not thread-safe.
var tapeMutex = sync.Mutex{}

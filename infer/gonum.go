package infer

import (
	"bitbucket.org/dtolpin/infergo/ad"
	"bitbucket.org/dtolpin/infergo/model"
	"sync"
)

// The autodiff tape is not re-entrant, we need a mutex.
var tapeMutex = sync.Mutex{}

// FuncGrad returns the function to minimize and the gradient,
// suitable as fields for gonum optimize.Problem, corresponding
// to maximization of the model's log-likelihood.
func FuncGrad(m model.Model) (
	Func func(x []float64) float64,
	Grad func(grad []float64, x []float64) []float64,
) {
	if ad.IsMTSafe() {
		// It is safe to run multiple differentiations in
		// parallel, no locking.

		Func = func(x []float64) float64 {
			ll := m.Observe(x)
			ad.Pop()
			return -ll
		}

		Grad = func(grad []float64, x []float64) []float64 {
			_, grad_ := m.Observe(x), ad.Gradient()
			if grad == nil {
				grad = grad_
			}
			for i := range grad_ {
				grad[i] = -grad_[i]
			}
			return grad
		}
	} else {
		// The tape must be locked.

		Func = func(x []float64) float64 {
			tapeMutex.Lock()
			defer tapeMutex.Unlock()
			ll := m.Observe(x)
			ad.Pop()
			return -ll
		}

		Grad = func(grad []float64, x []float64) []float64 {
			tapeMutex.Lock()
			defer tapeMutex.Unlock()
			_, grad_ := m.Observe(x), ad.Gradient()
			if grad == nil {
				grad = grad_
			}
			for i := range grad_ {
				grad[i] = -grad_[i]
			}
			return grad
		}
	}

	return Func, Grad
}

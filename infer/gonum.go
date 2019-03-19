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
	func(x []float64) float64,
	func(grad []float64, x []float64) []float64,
) {
	Func :=  func(x []float64) float64 {
		defer tapeMutex.Unlock()
		tapeMutex.Lock()
		ll := m.Observe(x)
		ad.Pop()
		return -ll
	}

	Grad := func(grad []float64, x []float64) []float64 {
		defer tapeMutex.Unlock()
		tapeMutex.Lock()
		_, grad_ := m.Observe(x), ad.Gradient()
		if grad == nil {
			grad = grad_
		}
		for i := range grad_ {
			grad[i] = -grad_[i]
		}
		return grad
	}

	return Func, Grad
}

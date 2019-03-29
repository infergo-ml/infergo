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
	func(x []float64) float64,
	func(grad []float64, x []float64) []float64,
) {
	Func := func(x []float64) float64 {
		ll := m.Observe(x)
		ad.Pop()
		return -ll
	}

	Grad := func(grad []float64, x []float64) []float64 {
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

// FuncGradM is a mutexed version of FuncGrad. Should be used
// when the inference algorithm is parallelized. The model must
// be immutable.
func FuncGradM(m model.Model) (
	func(x []float64) float64,
	func(grad []float64, x []float64) []float64,
) {
	Func := func(x []float64) float64 {
		tapeMutex.Lock()
		defer tapeMutex.Unlock()
		ll := m.Observe(x)
		ad.Pop()
		return -ll
	}

	Grad := func(grad []float64, x []float64) []float64 {
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

	return Func, Grad
}

// Tape mutex for FuncGradM
var tapeMutex = sync.Mutex{}

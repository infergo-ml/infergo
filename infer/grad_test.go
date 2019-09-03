package infer

// Testing gradient ascent algorithms.

import (
	"bitbucket.org/dtolpin/infergo/ad"
	"bitbucket.org/dtolpin/infergo/model"
	"math"
	"testing"
)

// A model that returns the constGrad gradient,
// for testing optimization algorithms.
type constGrad struct {
	grad []float64
}

func (m *constGrad) Observe(x []float64) float64 {
	ad.Setup(x)
	var ll float64
	ad.Assignment(&ll, ad.Value(0))
	for i := range x {
		ad.Assignment(&ll,
			ad.Arithmetic(ad.OpAdd, &ll,
				ad.Arithmetic(ad.OpMul, &m.grad[i], &x[i])))
	}
	return ad.Return(&ll)
}

// Just tests that the model indeed returns the
// constGrad gradient.
func TestModel(t *testing.T) {
	m := &constGrad{
		grad: []float64{1, 2},
	}
	m.Observe([]float64{3, 1})
	grad := model.Gradient(m)
	for i := range grad {
		if grad[i] != m.grad[i] {
			t.Fatalf("wrong gradient: got %v, want %v",
				grad, m.grad)
		}
	}
}

func TestMomentum(t *testing.T) {
	opt := &Momentum{
		Rate:  0.1,
		Decay: 0.5,
	}
	m := &constGrad{
		grad: []float64{1, 2},
	}
	x := []float64{0, 0}
	// Basic update
	xNext := []float64{0.1, 0.2}
	opt.Step(m, x)
	for i := range x {
		if math.Abs(xNext[i]-x[i]) > 1E-6 {
			t.Errorf("wrong first update: got x[%d] = %.6g, "+
				"want %.6g", i, x[i], xNext[i])
		}
	}
	// Update with decay
	xNext = []float64{0.15, 0.3}
	opt.Step(m, x)
	for i := range x {
		if math.Abs(xNext[i]-x[i]) > 1E-6 {
			t.Errorf("wrong second update (decay): "+
				"got x[%d] = %.6g, want %.6g",
				i, x[i], xNext[i])
		}
	}
	// Update with momentum
	opt.Gamma = 0.25
	xNext = []float64{0.1875, 0.375}
	opt.Step(m, x)
	for i := range x {
		if math.Abs(xNext[i]-x[i]) > 1E-6 {
			t.Errorf("wrong third update (momentum): "+
				"got x[%d] = %.6g, want %.6g",
				i, x[i], xNext[i])
		}
	}
}

func TestAdam(t *testing.T) {
	opt := &Adam{
		Rate: 0.1,
	}
	m := &constGrad{
		grad: []float64{1, 2},
	}
	x := []float64{0, 0}
	xNext := []float64{0.1, 0.1}
	opt.Step(m, x)
	for i := range x {
		if math.Abs(xNext[i]-x[i]) > 1E-6 {
			t.Errorf("wrong first update: got x[%d] = %.6g, "+
				"want %.6g", i, x[i], xNext[i])
		}
	}
	opt.Step(m, x)
	xNext = []float64{0.2, 0.2}
	for i := range x {
		if math.Abs(xNext[i]-x[i]) > 1E-6 {
			t.Errorf("wrong second update (decay): "+
				"got x[%d] = %.6g, want %.6g",
				i, x[i], xNext[i])
		}
	}
}

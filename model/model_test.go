package model

import (
	"bitbucket.org/dtolpin/infergo/ad"
	"reflect"
	"testing"
)

// A model with zero gradient.
type adModel struct{}

func (*adModel) Observe(x []float64) float64 {
	ad.Setup(x)
	return ad.Return(ad.Value(0))
}

// An elemental model with identity gradient.
type elModel struct{ grad []float64 }

func (m *elModel) Observe(x []float64) float64 {
	m.grad = x
	return 0.
}

func (m *elModel) Gradient() []float64 {
	return m.grad
}

func TestGradient(t *testing.T) {
	for i, c := range []struct {
		m    Model
		x    []float64
		grad []float64
	}{
		{
			&adModel{},
			[]float64{1., 2.},
			[]float64{0., 0.},
		},
		{
			&elModel{},
			[]float64{2., 1.},
			[]float64{2., 1.},
		},
	} {
		c.m.Observe(c.x)
		grad := Gradient(c.m)
		for j := range grad {
			if grad[j] != c.grad[j] {
				t.Fatalf("%d: wrong gradient for %T: got %v, want %v",
					i, c.m, grad, c.grad)
			}
		}
	}
}

func TestShift(t *testing.T) {
	// kicking tyres
	for i, c := range []struct {
		x0    []float64
		n     int
		y, x1 []float64
	}{
		{
			[]float64{1, 2},
			2,
			[]float64{1, 2}, []float64{},
		},
		{
			[]float64{1, 2},
			1,
			[]float64{1}, []float64{2},
		},
		{
			[]float64{1, 2, 3},
			0,
			[]float64{}, []float64{1, 2, 3},
		},
	} {
		x := c.x0
		y := Shift(&x, c.n)
		if !reflect.DeepEqual(y, c.y) {
			t.Errorf("%d: wrong retrieved parameters for %+v: "+
				"got %v, want %v",
				i, c, y, c.y)
		}
	}
}

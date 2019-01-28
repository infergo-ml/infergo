package model

import (
	"reflect"
	"testing"
)

func TestShift(t *testing.T) {
	// kicking tyres
	for i, c := range []struct {
		x0 []float64
		n int
		y, x1 []float64
	} {
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

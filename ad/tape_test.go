package ad

// Testing the tape

import (
	"reflect"
	"testing"
	"math"
)

// dfdx differentiates the function passed in
// and returns the gradient.
func dfdx(x []float64, f func(x []float64)) []float64 {
	Enter(len(x))
	for i := 0; i != len(x); i++ {
		Variable(&x[i])
	}
	f(x)
	return Gradient()
}

func TestPrimitive(t *testing.T) {
	for _, c := range []struct {
		s string
		f func(x []float64)
		v [][][]float64
	}{
		{"v = u",
			func(x []float64) {
				Assignment(Variable(Constant(x[0])), &x[0])
			},
			[][][]float64{
				{{0.}, {1.}},
				{{1.}, {1.}}}},
		{"v = v",
			func(x []float64) {
				Assignment(&x[0], &x[0])
			},
			[][][]float64{
				{{0.}, {1.}},
				{{1.}, {1.}}}},
		{"u + w",
			func(x []float64) {
				Arithmetic(OpAdd,
					Variable(Constant(x[0] + x[1])), &x[0], &x[1])
			},
			[][][]float64{
				{{0., 0.}, {1., 1.}},
				{{3., 5.}, {1., 1.}}}},
		{"u + u",
			func(x []float64) {
				Arithmetic(OpAdd,
					Variable(Constant(x[0] + x[0])), &x[0], &x[0])
			},
			[][][]float64{
				{{0.}, {2.}},
				{{1.}, {2.}}}},
		{"u - v",
			func(x []float64) {
				Arithmetic(OpSub,
					Variable(Constant(x[0] - x[1])), &x[0], &x[1])
			},
			[][][]float64{
				{{0., 0.}, {1., -1.}},
				{{1., 1.}, {1., -1.}}}},
		{"u - u",
			func(x []float64) {
				Arithmetic(OpSub,
					Variable(Constant(x[0] - x[0])), &x[0], &x[0])
			},
			[][][]float64{
				{{0.}, {0.}},
				{{1.}, {0.}}}},
		{"u * w",
			func(x []float64) {
				Arithmetic(OpMul,
					Variable(Constant(x[0] * x[1])), &x[0], &x[1])
			},
			[][][]float64{
				{{0., 0.}, {0., 0.}},
				{{2., 3.}, {3., 2.}}}},
		{"u * u",
			func(x []float64) {
				Arithmetic(OpMul,
					Variable(Constant(x[0] * x[0])), &x[0], &x[0])
			},
			[][][]float64{
				{{0.}, {0.}},
				{{1.}, {2.}}}},
		{"u / w",
			func(x []float64) {
				Arithmetic(OpDiv,
					Variable(Constant(x[0] / x[1])), &x[0], &x[1])
			},
			[][][]float64{
				{{0., 1.}, {1., 0.}},
				{{2., 4.}, {0.25, - 0.125}}}},
		{"u / u",
			func(x []float64) {
				Arithmetic(OpDiv,
				    Variable(Constant(x[0] / x[0])), &x[0], &x[0])
			},
			[][][]float64{
				{{1.}, {0.}},
				{{2.}, {0.}}}},
		{"sqrt(u)",
			func(x []float64) {
				Elemental(math.Sqrt,
					Variable(Constant(math.Sqrt(x[0]))), &x[0])
			},
			[][][]float64{
				{{0.25}, {1.}},
				{{1.}, {0.5}},
				{{4.}, {0.25}}}},
		{"log(u)",
			func(x []float64) {
				Elemental(math.Log,
					Variable(Constant(math.Log(x[0]))), &x[0])
			},
			[][][]float64{
				{{1.}, {1.}},
				{{2.}, {0.5}}}},
		{"exp(u)",
			func(x []float64) {
				Elemental(math.Exp,
					Variable(Constant(math.Exp(x[0]))), &x[0])
			},
			[][][]float64{
				{{0.}, {1.}},
				{{1.}, {math.E}}}},
		{"cos(u)",
			func(x []float64) {
				Elemental(math.Cos,
					Variable(Constant(math.Cos(x[0]))), &x[0])
			},
			[][][]float64{
				{{0.}, {0.}},
				{{1.}, {- math.Sin(1.)}}}},
		{"sin(u)",
			func(x []float64) {
				Elemental(math.Sin,
					Variable(Constant(math.Sin(x[0]))), &x[0])
			},
			[][][]float64{
				{{0.}, {1.}},
				{{1.}, {math.Cos(1.)}}}},
	} {
		for _, v := range c.v {
			g := dfdx(v[0], c.f)
			if !reflect.DeepEqual(g, v[1]) {
				t.Errorf("%s, x=%v: g=%v, wanted g=%v",
					c.s, v[0], g, v[1])
			}
		}
	}
}

func TestComposite(t *testing.T) {
}

package ad

// Testing the tape

import (
	"math"
	"reflect"
	"testing"
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

// testcase defines a test of a single expression on
// several inputs.
type testcase struct {
	s string
	f func(x []float64)
	v [][][]float64
}

// runsuite evaluates a sequence of test cases.
func runsuite(t *testing.T, suite []testcase) {
	for _, c := range suite {
		for _, v := range c.v {
			g := dfdx(v[0], c.f)
			if !reflect.DeepEqual(g, v[1]) {
				t.Errorf("%s, x=%v: g=%v, wanted g=%v",
					c.s, v[0], g, v[1])
			}
		}
	}
}

func TestPrimitive(t *testing.T) {
	runsuite(t, []testcase{
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
					Variable(Constant(x[0]+x[1])), &x[0], &x[1])
			},
			[][][]float64{
				{{0., 0.}, {1., 1.}},
				{{3., 5.}, {1., 1.}}}},
		{"u + u",
			func(x []float64) {
				Arithmetic(OpAdd,
					Variable(Constant(x[0]+x[0])), &x[0], &x[0])
			},
			[][][]float64{
				{{0.}, {2.}},
				{{1.}, {2.}}}},
		{"u - v",
			func(x []float64) {
				Arithmetic(OpSub,
					Variable(Constant(x[0]-x[1])), &x[0], &x[1])
			},
			[][][]float64{
				{{0., 0.}, {1., -1.}},
				{{1., 1.}, {1., -1.}}}},
		{"u - u",
			func(x []float64) {
				Arithmetic(OpSub,
					Variable(Constant(x[0]-x[0])), &x[0], &x[0])
			},
			[][][]float64{
				{{0.}, {0.}},
				{{1.}, {0.}}}},
		{"u * w",
			func(x []float64) {
				Arithmetic(OpMul,
					Variable(Constant(x[0]*x[1])), &x[0], &x[1])
			},
			[][][]float64{
				{{0., 0.}, {0., 0.}},
				{{2., 3.}, {3., 2.}}}},
		{"u * u",
			func(x []float64) {
				Arithmetic(OpMul,
					Variable(Constant(x[0]*x[0])), &x[0], &x[0])
			},
			[][][]float64{
				{{0.}, {0.}},
				{{1.}, {2.}}}},
		{"u / w",
			func(x []float64) {
				Arithmetic(OpDiv,
					Variable(Constant(x[0]/x[1])), &x[0], &x[1])
			},
			[][][]float64{
				{{0., 1.}, {1., 0.}},
				{{2., 4.}, {0.25, -0.125}}}},
		{"u / u",
			func(x []float64) {
				Arithmetic(OpDiv,
					Variable(Constant(x[0]/x[0])), &x[0], &x[0])
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
				{{1.}, {-math.Sin(1.)}}}},
		{"sin(u)",
			func(x []float64) {
				Elemental(math.Sin,
					Variable(Constant(math.Sin(x[0]))), &x[0])
			},
			[][][]float64{
				{{0.}, {1.}},
				{{1.}, {math.Cos(1.)}}}},
	})
}

func TestComposite(t *testing.T) {
	runsuite(t, []testcase{
		{"v = u * u + w * w",
			func(x []float64) {
				a := x[0] * x[0]
				b := x[1] * x[1]
				c := a + b
				Arithmetic(OpAdd,
					Variable(Constant(c)),
					Arithmetic(OpMul,
						Variable(Constant(a)), &x[0], &x[0]),
					Arithmetic(OpMul,
						Variable(Constant(b)), &x[1], &x[1]))
			},
			[][][]float64{
				{{0., 0.}, {0., 0.}},
				{{1., 1.}, {2., 2.}},
				{{2., 3.}, {4., 6.}}}},
		{"v = (u + w) * (u + w)",
			func(x []float64) {
				a := x[0] + x[1]
				b := x[0] + x[1]
				c := a * b
				Arithmetic(OpMul,
					Variable(Constant(c)),
					Arithmetic(OpAdd,
						Variable(Constant(a)), &x[0], &x[1]),
					Arithmetic(OpAdd,
						Variable(Constant(b)), &x[0], &x[1]))
			},
			[][][]float64{
				{{0., 0.}, {0., 0.}},
				{{1., 1.}, {4., 4.}},
				{{2., 3.}, {10., 10.}}}},
		{"v = sin(u * w)",
			func(x []float64) {
				a := x[0] * x[1]
				b := math.Sin(a)
				Elemental(math.Sin,
					Variable(Constant(b)),
					Arithmetic(OpMul,
						Variable(Constant(a)), &x[0], &x[1]))
			},
			[][][]float64{
				{{0., 0.}, {0., 0.}},
				{{1., math.Pi}, {-math.Pi, -1.}},
				{{math.Pi, 1.}, {-1., -math.Pi}}}},
	})
}

func TestAssignment(t *testing.T) {
	runsuite(t, []testcase{
		{"v = sin(u * w); v1 = v",
			func(x []float64) {
				a := x[0] * x[1]
				b := math.Sin(a)
				Assignment(Variable(Constant(-1.)),
					Elemental(math.Sin,
						Variable(Constant(b)),
						Arithmetic(OpMul,
							Variable(Constant(a)), &x[0], &x[1])))
			},
			[][][]float64{
				{{0., 0.}, {0., 0.}},
				{{1., math.Pi}, {-math.Pi, -1.}},
				{{math.Pi, 1.}, {-1., -math.Pi}}}},
		{"u = 2.; v = u*u",
			func(x []float64) {
				Assignment(&x[0], Variable(Constant(2)))
				x[0] = 2.
				b := x[0] * x[0]
				Arithmetic(OpMul, Variable(Constant(b)), &x[0], &x[0])
			},
			[][][]float64{
				{{0.}, {0.}},
				{{3.}, {0.}}}},
		{"u = u; v = u*u",
			func(x []float64) {
				Assignment(&x[0], &x[0])
				x[0] = x[0]
				b := x[0] * x[0]
				Arithmetic(OpMul, Variable(Constant(b)), &x[0], &x[0])
			},
			[][][]float64{
				{{0.}, {0.}},
				{{3.}, {6.}}}},
	})
}

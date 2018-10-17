package ad

// Testing the tape

import (
	"math"
	"math/rand"
	"reflect"
	"testing"
	"time"
)

// Intermediate variables are initialized to random
// values for testing.
func init() {
	rand.Seed(time.Now().UnixNano())
}

// dfdx differentiates the function passed in
// and returns the gradient.
func dfdx(x []float64, f func(x []float64)) []float64 {
	Enter(len(x))
	for i := 0; i != len(x); i++ {
		Place(&x[i])
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

// placeholder returns an uninitialized placeholder
func placeholder() *float64 {
	return Place(Value(rand.ExpFloat64()))
}

func TestPrimitive(t *testing.T) {
	runsuite(t, []testcase{
		{"x = y",
			func(x []float64) {
				Assignment(placeholder(), &x[0])
			},
			[][][]float64{
				{{0.}, {1.}},
				{{1.}, {1.}}}},
		{"x = x",
			func(x []float64) {
				Assignment(&x[0], &x[0])
			},
			[][][]float64{
				{{0.}, {1.}},
				{{1.}, {1.}}}},
		{"x + y",
			func(x []float64) {
				Arithmetic(OpAdd, placeholder(), &x[0], &x[1])
			},
			[][][]float64{
				{{0., 0.}, {1., 1.}},
				{{3., 5.}, {1., 1.}}}},
		{"x + x",
			func(x []float64) {
				Arithmetic(OpAdd, placeholder(), &x[0], &x[0])
			},
			[][][]float64{
				{{0.}, {2.}},
				{{1.}, {2.}}}},
		{"x - z",
			func(x []float64) {
				Arithmetic(OpSub, placeholder(), &x[0], &x[1])
			},
			[][][]float64{
				{{0., 0.}, {1., -1.}},
				{{1., 1.}, {1., -1.}}}},
		{"x - x",
			func(x []float64) {
				Arithmetic(OpSub, placeholder(), &x[0], &x[0])
			},
			[][][]float64{
				{{0.}, {0.}},
				{{1.}, {0.}}}},
		{"x * y",
			func(x []float64) {
				Arithmetic(OpMul, placeholder(), &x[0], &x[1])
			},
			[][][]float64{
				{{0., 0.}, {0., 0.}},
				{{2., 3.}, {3., 2.}}}},
		{"x * x",
			func(x []float64) {
				Arithmetic(OpMul, placeholder(), &x[0], &x[0])
			},
			[][][]float64{
				{{0.}, {0.}},
				{{1.}, {2.}}}},
		{"x / y",
			func(x []float64) {
				Arithmetic(OpDiv, placeholder(), &x[0], &x[1])
			},
			[][][]float64{
				{{0., 1.}, {1., 0.}},
				{{2., 4.}, {0.25, -0.125}}}},
		{"x / x",
			func(x []float64) {
				Arithmetic(OpDiv, placeholder(), &x[0], &x[0])
			},
			[][][]float64{
				{{1.}, {0.}},
				{{2.}, {0.}}}},
		{"sqrt(x)",
			func(x []float64) {
				Elemental(math.Sqrt, placeholder(), &x[0])
			},
			[][][]float64{
				{{0.25}, {1.}},
				{{1.}, {0.5}},
				{{4.}, {0.25}}}},
		{"log(x)",
			func(x []float64) {
				Elemental(math.Log, placeholder(), &x[0])
			},
			[][][]float64{
				{{1.}, {1.}},
				{{2.}, {0.5}}}},
		{"exp(x)",
			func(x []float64) {
				Elemental(math.Exp, placeholder(), &x[0])
			},
			[][][]float64{
				{{0.}, {1.}},
				{{1.}, {math.E}}}},
		{"cos(x)",
			func(x []float64) {
				Elemental(math.Cos, placeholder(), &x[0])
			},
			[][][]float64{
				{{0.}, {0.}},
				{{1.}, {-math.Sin(1.)}}}},
		{"sin(x)",
			func(x []float64) {
				Elemental(math.Sin, placeholder(), &x[0])
			},
			[][][]float64{
				{{0.}, {1.}},
				{{1.}, {math.Cos(1.)}}}},
	})
}

func TestComposite(t *testing.T) {
	runsuite(t, []testcase{
		{"x * x + y * y",
			func(x []float64) {
				Arithmetic(OpAdd, placeholder(),
					Arithmetic(OpMul, placeholder(), &x[0], &x[0]),
					Arithmetic(OpMul, placeholder(), &x[1], &x[1]))
			},
			[][][]float64{
				{{0., 0.}, {0., 0.}},
				{{1., 1.}, {2., 2.}},
				{{2., 3.}, {4., 6.}}}},
		{"(x + y) * (x + y)",
			func(x []float64) {
				Arithmetic(OpMul, placeholder(),
					Arithmetic(OpAdd, placeholder(), &x[0], &x[1]),
					Arithmetic(OpAdd, placeholder(), &x[0], &x[1]))
			},
			[][][]float64{
				{{0., 0.}, {0., 0.}},
				{{1., 1.}, {4., 4.}},
				{{2., 3.}, {10., 10.}}}},
		{"sin(x * y)",
			func(x []float64) {
				Elemental(math.Sin, placeholder(),
					Arithmetic(OpMul, placeholder(), &x[0], &x[1]))
			},
			[][][]float64{
				{{0., 0.}, {0., 0.}},
				{{1., math.Pi}, {-math.Pi, -1.}},
				{{math.Pi, 1.}, {-1., -math.Pi}}}},
	})
}

func TestAssignment(t *testing.T) {
	runsuite(t, []testcase{
		{"z = sin(x * y); v1 = z",
			func(x []float64) {
				Assignment(placeholder(),
					Elemental(math.Sin, placeholder(),
						Arithmetic(OpMul,
							placeholder(), &x[0], &x[1])))
			},
			[][][]float64{
				{{0., 0.}, {0., 0.}},
				{{1., math.Pi}, {-math.Pi, -1.}},
				{{math.Pi, 1.}, {-1., -math.Pi}}}},
		{"x = 2.; z = x * x",
			func(x []float64) {
				Assignment(&x[0], Place(Value(2.)))
				Arithmetic(OpMul, placeholder(), &x[0], &x[0])
			},
			[][][]float64{
				{{0.}, {0.}},
				{{3.}, {0.}}}},
		{"x = x; z = x * x",
			func(x []float64) {
				Assignment(&x[0], &x[0])
				Arithmetic(OpMul, placeholder(), &x[0], &x[0])
			},
			[][][]float64{
				{{0.}, {0.}},
				{{3.}, {6.}}}},
	})
}

// elementals to check calling with different signatures
func twoArgElemental(a, b float64) float64 {
	return a * b
}

func threeArgElemental(a, b, c float64) float64 {
	return a + b + c
}

func variadicElemental(a ...float64) float64 {
	return a[0] - a[1]
}

func init() {
	RegisterElemental(twoArgElemental,
		func(v float64, a ...float64) []float64 {
			return []float64{a[1], a[0]}
		})
	RegisterElemental(threeArgElemental,
		func(v float64, a ...float64) []float64 {
			return []float64{1., 1., 1.}
		})
	RegisterElemental(variadicElemental,
		func(v float64, a ...float64) []float64 {
			return []float64{1., -1.}
		})
}

func TestElemental(t *testing.T) {
	runsuite(t, []testcase{
		{"twoArgElemental(x, y)",
			func(x []float64) {
				Elemental(twoArgElemental,
					placeholder(), &x[0], &x[1])
			},
			[][][]float64{
				{{0., 0.}, {0., 0.}},
				{{1., 2.}, {2., 1.}}}},
		{"threeArgElemental(x, y, t)",
			func(x []float64) {
				Elemental(threeArgElemental,
					placeholder(), &x[0], &x[1], &x[2])
			},
			[][][]float64{
				{{0., 0., 0.}, {1., 1., 1.}},
				{{1., 2., 3.}, {1., 1., 1.}}}},
		{"variadicElemental(x, y)",
			func(x []float64) {
				Elemental(variadicElemental,
					placeholder(), &x[0], &x[1])
			},
			[][][]float64{
				{{0., 0.}, {1., -1.}},
				{{1., 2.}, {1., -1.}}}},
	})
}

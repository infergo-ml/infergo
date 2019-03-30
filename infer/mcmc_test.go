package infer

// Testing MCMC algorithms.

import (
	"bitbucket.org/dtolpin/infergo/ad"
	. "bitbucket.org/dtolpin/infergo/dist/ad"
	"math"
	"math/rand"
	"testing"
	"time"
)

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

// Unit tests for helpers

func TestEnergy(t *testing.T) {
	for _, c := range []struct {
		l float64
		r []float64
		e float64
	}{
		{1, []float64{0}, 1},
		{1, []float64{1, 3}, -4},
	} {
		if e := energy(c.l, c.r); math.Abs(e-c.e) > 1E-6 {
			t.Errorf("incorrect energy for l=%v, r=%v: "+
				"got=%.6g, want=%.6g", c.l, c.r, e, c.e)
		}
	}
}

func TestLeapfrog(t *testing.T) {
	// constGrad is defined in infer/infer_test.go
	m := &constGrad{
		grad: []float64{0.5, 1.5},
	}
	x, r, eps := []float64{0, 0}, []float64{1, -1}, 0.5
	m.Observe(x)
	grad := ad.Gradient()
	_, grad = leapfrog(m, grad, x, r, eps)
	xNext, rNext := []float64{0.5625, -0.3125}, []float64{1.25, -0.25}
	for i := range x {
		if math.Abs(x[i]-xNext[i]) > 1E-6 {
			t.Errorf("wrong leapfrog step: got x[%d] = %.6g, "+
				"want %.6g", i, x[i], xNext[i])
		}
	}
	for i := range x {
		if math.Abs(r[i]-rNext[i]) > 1E-6 {
			t.Errorf("wrong leapfrog step: got r[%d] = %.6g, "+
				"want %.6g", i, r[i], rNext[i])
		}
	}
}

// Testing samplers

// A model for testing. The model infers parameters of the
// Normal distribution given a data set. The model parameters
// are mean and log stddev. The model does not use Logps to test
// a more general code than a single method call.
type testModel struct {
	data []float64
}

func (m *testModel) Observe(x []float64) float64 {
	ad.Setup(x)
	var ll float64
	ad.Assignment(&ll, ad.Value(0))
	var stddev float64
	ad.Assignment(&stddev, ad.Elemental(math.Exp, &x[1]))
	for i := range m.data {
		ad.Assignment(&ll, ad.Arithmetic(ad.OpAdd,
			&ll,
			ad.Call(func(_ []float64) {
				Normal.Logp(0, 0, 0)
			}, 3, &x[0], &stddev, &m.data[i])))
	}
	return ad.Return(&ll)
}

// A small data set for testing.
var (
	testData             []float64
	testMean, testStddev float64
)

func init() {
	testData = []float64{
		-0.854, 1.067, -1.220, 0.818, -0.749,
		0.805, 1.443, 1.069, 1.426, 0.308}
	s, s2 := 0., 0.
	for _, y := range testData {
		s += y
		s2 += y * y
	}
	n := float64(len(testData))
	testMean = s / n
	testStddev = math.Sqrt(s2/n - testMean*testMean)
}

// repeatedly runs the test function n times and returns true if
// the test returned true at least once, false otherwise. Used
// for statistical testing of stochastic algorithms.
func repeatedly(
	nattempts int,
	test func() bool,
	breakEarly bool,
) bool {
	succeeded := false
	for i := 0; i != nattempts; i++ {
		if test() {
			succeeded = true
			if breakEarly {
				break
			}
		}
	}
	return succeeded
}

// inferMeanStddev infers the mean and standard deviation of
// the test data set.
func inferMeanStddev(
	sampler MCMC, niter int,
) (mean, stddev float64) {
	m := &testModel{testData}
	x := []float64{0.1 * rand.NormFloat64(), 0.1 * rand.NormFloat64()}
	samples := make(chan []float64)
	sampler.Sample(m, x, samples)
	// Burn
	for i := 0; i != niter; i++ {
		<-samples
	}
	// Collect after burn-in
	n := 0.
	for i := 0; i != niter; i++ {
		x := <-samples
		if len(x) == 0 {
			break
		}
		mean += x[0]
		stddev += math.Exp(x[1])
		n++
	}
	sampler.Stop()
	mean /= n
	stddev /= n
	return mean, stddev
}

func TestUTurn(t *testing.T) {
	for _, c := range []struct {
		xl, xr, r []float64
		uturn     bool
	}{
		{
			[]float64{-1, 0}, []float64{1, 0},
			[]float64{1, 0},
			false,
		},
		{
			[]float64{-1, 0}, []float64{-1, 0},
			[]float64{0, 1},
			false,
		},
		{
			[]float64{1, 0}, []float64{0, 1},
			[]float64{1, 0},
			true,
		},
	} {
		if c.uturn {
			if !uTurn(c.xl, c.xr, c.r) {
				t.Errorf("missed uturn: %+v", c)
			}
		} else {
			if uTurn(c.xl, c.xr, c.r) {
				t.Errorf("false uturn: %+v", c)
			}
		}
	}
}

// Basic convergence of MCMC samplers. Empirical mean and stddev
// should be around the inferred mean and stddev.
func TestSamplers(t *testing.T) {
	nattempts := 10
	niter := 100
	prec := 1E-1
	for _, c := range []struct {
		sampler func() MCMC
	}{
		{
			func() MCMC {
				return &HMC{
					L:   5,
					Eps: 0.1,
				}
			},
		},
		{
			func() MCMC {
				return &NUTS{
					Eps: 0.1,
				}
			},
		},
	} {
		if !repeatedly(nattempts,
			func() bool {
				mean, stddev := inferMeanStddev(c.sampler(), niter)
				return math.Abs((mean-testMean)/
					(mean+testMean)) <= prec &&
					math.Abs((stddev-testStddev)/
						(stddev+testStddev)) <= prec
			},
			true) {
			t.Errorf("%T did not converge", c.sampler())
		}
	}
}

func TestNUTSDepth(t *testing.T) {
	nuts := &NUTS{}
	for _, c := range []struct {
		depths    []int
		meanDepth float64
	}{
		{[]int{}, 0},
		{[]int{0}, 0},
		{[]int{1}, 1},
		{[]int{2, 2}, 2},
		{[]int{2, 1}, 1.5},
		{[]int{0, 1, 2}, 1},
	} {
		nuts.Depth = nil
		for _, depth := range c.depths {
			nuts.updateDepth(depth)
		}
		if meanDepth := nuts.MeanDepth(); meanDepth != c.meanDepth {
			t.Errorf("wrong average depth for %v: "+
				"got %.4f, want %.4g",
				c.depths, meanDepth, c.meanDepth)
		}
	}
}

const BenchmarkNiter = 100

func init() {
}

func BenchmarkHmcL10Eps01(b *testing.B) {
	for i := 0; i != b.N; i++ {
		inferMeanStddev(
			&HMC{
				L:   10,
				Eps: 0.1,
			}, BenchmarkNiter)
	}
}

func BenchmarkHmcL20Eps005(b *testing.B) {
	for i := 0; i != b.N; i++ {
		inferMeanStddev(
			&HMC{
				L:   20,
				Eps: 0.05,
			}, BenchmarkNiter)
	}
}

func BenchmarkNutsEps01(b *testing.B) {
	for i := 0; i != b.N; i++ {
		inferMeanStddev(
			&NUTS{
				Eps: 0.1,
			}, BenchmarkNiter)
	}
}

func BenchmarkNutsEps005(b *testing.B) {
	for i := 0; i != b.N; i++ {
		inferMeanStddev(
			&NUTS{
				Eps: 0.05,
			}, BenchmarkNiter)
	}
}

func BenchmarkHmcL10Eps01MT(b *testing.B) {
	// Just turning on multithreading safety.
	ad.MTSafeOn()

	BenchmarkHmcL10Eps01(b)
	ad.DropAllTapes()
}

func BenchmarkHmcL20Eps005MT(b *testing.B) {
	BenchmarkHmcL20Eps005(b)
	ad.DropAllTapes()
}

func BenchmarkNutsEps01MT(b *testing.B) {
	BenchmarkNutsEps01(b)
	ad.DropAllTapes()
}

func BenchmarkNutsEps005MT(b *testing.B) {
	BenchmarkNutsEps005(b)
	ad.DropAllTapes()
}

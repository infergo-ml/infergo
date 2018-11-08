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
		{1., []float64{0.}, 1},
		{1., []float64{1., 3.}, 6},
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
	x, r, eps := []float64{0., 0.}, []float64{1., -1.}, 0.5
	m.Observe(x)
	grad := ad.Gradient()
	leapfrog(m, &grad, x, r, eps)
	xNext, rNext := []float64{0.5625, -0.3125}, []float64{1.25, -0.25}
	for i := 0; i != len(x); i++ {
		if math.Abs(x[i]-xNext[i]) > 1E-6 {
			t.Errorf("wrong leapfrog step: got x[%d] = %.6g, "+
				"want %.6g", i, x[i], xNext[i])
		}
	}
	for i := 0; i != len(x); i++ {
		if math.Abs(r[i]-rNext[i]) > 1E-6 {
			t.Errorf("wrong leapfrog step: got r[%d] = %.6g, "+
				"want %.6g", i, r[i], rNext[i])
		}
	}
}

func TestUTurn(t *testing.T) {
	for _, c := range []struct {
		xl, rl, xr, rr []float64
		uturn          bool
	}{
		{
			[]float64{-1., 0}, []float64{0., 1.},
			[]float64{1., 0.}, []float64{1., 0.},
			false,
		},
		{
			[]float64{-1., 0}, []float64{0., 1.},
			[]float64{-1., 0.}, []float64{1., 0.},
			false,
		},
		{
			[]float64{1., 0}, []float64{0., 1.},
			[]float64{0., 1.}, []float64{1., 0.},
			true,
		},
		{
			[]float64{1., 0}, []float64{0., 1.},
			[]float64{0., 1.}, []float64{-1., 0.},
			false,
		},
	} {
		if c.uturn {
			if !uTurn(c.xl, c.rl, c.xr, c.rr) {
				t.Errorf("missed uturn: %+v", c)
			}
		} else {
			if uTurn(c.xl, c.rl, c.xr, c.rr) {
				t.Errorf("false uturn: %+v", c)
			}
		}
	}
}

// Testing samplers

// A model for testing. The model infers parameters
// of the Normal distribution given a data set. The
// model parameters are mean and logvariance.
type testModel struct {
	data []float64
}

func (m *testModel) Observe(x []float64) float64 {
	ad.Setup(x)
	var ll float64
	ad.Assignment(&ll, ad.Value(0.))
	for i := 0; i != len(m.data); i++ {
		ad.Assignment(&ll, ad.Arithmetic(ad.OpAdd,
			&ll,
			ad.Call(func(_vararg []float64) {
				Normal.Logp(0, 0, 0)
			}, 3, &m.data[i], &x[0], &x[1])))
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
	for i := 0; i != len(testData); i++ {
		s += testData[i]
		s2 += testData[i] * testData[i]
	}
	n := float64(len(testData))
	testMean = s / n
	testStddev = math.Sqrt(s2/n - testMean*testMean)
}

// repeatedly runs the thunk n times and returns true if the
// test returned true at least once, false otherwise. Used for
// statistical testing of stochastic algorithms.
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
	x := []float64{rand.NormFloat64(), rand.NormFloat64()}
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
		stddev += math.Exp(0.5 * x[1])
		n++
	}
	sampler.Stop()
	mean /= n
	stddev /= n
	return mean, stddev
}

// Test MCMC samplers for basic convergence. Empirical mean and
// stddev should be around the inferred mean and stddev.
func TestSamplers(t *testing.T) {
	for _, c := range []struct {
		sampler          func() MCMC
		nattempts, niter int
		prec             float64
	}{
		{
			func() MCMC {
				return &HMC{
					L:   10,
					Eps: 0.05,
				}
			},
			3, 100,
			1E-2,
		},
		{
			func() MCMC {
				return &NUTS{
					Eps: 0.05,
				}
			},
			3, 100,
			1E-2,
		},
	} {
		if !repeatedly(c.niter,
			func() bool {
				mean, stddev := inferMeanStddev(c.sampler(), c.niter)
				return math.Abs(mean-testMean) <= c.prec &&
					math.Abs(stddev-testStddev) <= c.prec
			},
			true) {
			t.Errorf("%T did not converge", c.sampler())
		}
	}
}

func BenchmarkHmcL10Eps01(b *testing.B) {
    for i := 0; i != b.N; i++ {
        inferMeanStddev(
            &HMC {
                L: 10,
                Eps: 0.1,
            }, 100)
    }
}

func BenchmarkHmcL20Eps005(b *testing.B) {
    for i := 0; i != b.N; i++ {
        inferMeanStddev(
            &HMC {
                L: 20,
                Eps: 0.05,
            }, 100)
    }
}

func BenchmarkNutsEps01(b *testing.B) {
    for i := 0; i != b.N; i++ {
        inferMeanStddev(
            &NUTS {
                Eps: 0.1,
            }, 100)
    }
}

func BenchmarkNutsEps005(b *testing.B) {
    for i := 0; i != b.N; i++ {
        inferMeanStddev(
            &NUTS {
                Eps: 0.05,
            }, 100)
    }
}

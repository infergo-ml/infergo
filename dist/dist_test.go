package dist

// Testing distribution models.

import (
	"math"
	"testing"
)

func TestNormal(t *testing.T) {
	for _, c := range []struct {
		mean, stddev float64
		y            []float64
		ll           float64
	}{
		{0., 1., []float64{0.}, -0.9189385332046727},
		{1., 2., []float64{2.}, -1.737085713764618},
		{0., 1., []float64{-1., 0.}, -2.3378770664093453},
	} {
		logv := 2 * math.Log(c.stddev)
		ll := Normal.Logps(c.mean, logv, c.y...)
		if math.Abs(ll-c.ll) > 1E-6 {
			t.Errorf("Wrong logpdf of Normal(%.v|%.v, %.v): "+
				"got %.4g, want %.4g",
				c.y, c.mean, c.stddev, ll, c.ll)
		}
		llo := Normal.Observe(append([]float64{c.mean, logv}, c.y...))
		if math.Abs(ll-llo) > 1E-6 {
			t.Errorf("Wrong result of Observe([%.4g, %.4g, %v...]): "+
				"got %.4g, want %.4g",
				c.mean, logv, c.y, llo, ll)
		}
		if len(c.y) == 1 {
			ll1 := Normal.Logp(c.mean, logv, c.y[0])
			if math.Abs(ll-ll1) > 1E-6 {
				t.Errorf("Wrong result of Logp(%.4g, %.4g, %.4g): "+
					"got %.4g, want %.4g",
					c.mean, logv, c.y[0], ll1, ll)
			}
		}
	}
}

func TestExpon(t *testing.T) {
	for _, c := range []struct {
		lambda float64
		y      []float64
		ll     float64
	}{
		{1., []float64{1.}, -1},
		{2., []float64{2.}, -3.3068528194400546},
		{1., []float64{1., 2.}, -3},
	} {
		logl := math.Log(c.lambda)
		ll := Expon.Logps(logl, c.y...)
		if math.Abs(ll-c.ll) > 1E-6 {
			t.Errorf("Wrong logpdf of Expon(%.v|%.v): "+
				"got %.4g, want %.4g",
				c.y, c.lambda, ll, c.ll)
		}
		llo := Expon.Observe(append([]float64{logl}, c.y...))
		if math.Abs(ll-llo) > 1E-6 {
			t.Errorf("Wrong result of Observe([%.4g, %v...]): "+
				"got %.4g, want %.4g",
				logl, c.y, llo, ll)
		}
		if len(c.y) == 1 {
			ll1 := Expon.Logp(logl, c.y[0])
			if math.Abs(ll-ll1) > 1E-6 {
				t.Errorf("Wrong result of Logp(%.4g, %.4g): "+
					"got %.4g, want %.4g",
					logl, c.y[0], ll1, ll)
			}
		}
	}
}

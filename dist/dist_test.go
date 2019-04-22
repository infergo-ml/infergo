package dist

// Testing distribution models.

import (
	"math"
	"testing"
)

func TestNormal(t *testing.T) {
	for _, c := range []struct {
		mu, sigma float64
		y         []float64
		ll        float64
	}{
		{0., 1., []float64{0.}, -0.9189385332046727},
		{1., 2., []float64{2.}, -1.737085713764618},
		{0., 1., []float64{-1., 0.}, -2.3378770664093453},
	} {
		ll := Normal.Logps(c.mu, c.sigma, c.y...)
		if math.Abs(ll-c.ll) > 1E-6 {
			t.Errorf("Wrong logpdf of Normal(%.v|%.v, %.v): "+
				"got %.4g, want %.4g",
				c.y, c.mu, c.sigma, ll, c.ll)
		}
		llo := Normal.Observe(append([]float64{c.mu, c.sigma}, c.y...))
		if math.Abs(ll-llo) > 1E-6 {
			t.Errorf("Wrong result of Observe([%.4g, %.4g, %v...]): "+
				"got %.4g, want %.4g",
				c.mu, c.sigma, c.y, llo, ll)
		}
		if len(c.y) == 1 {
			ll1 := Normal.Logp(c.mu, c.sigma, c.y[0])
			if math.Abs(ll-ll1) > 1E-6 {
				t.Errorf("Wrong result of Logp(%.4g, %.4g, %.4g): "+
					"got %.4g, want %.4g",
					c.mu, c.sigma, c.y[0], ll1, ll)
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
		ll := Expon.Logps(c.lambda, c.y...)
		if math.Abs(ll-c.ll) > 1E-6 {
			t.Errorf("Wrong logpdf of Expon(%.v|%.v): "+
				"got %.4g, want %.4g",
				c.y, c.lambda, ll, c.ll)
		}
		llo := Expon.Observe(append([]float64{c.lambda}, c.y...))
		if math.Abs(ll-llo) > 1E-6 {
			t.Errorf("Wrong result of Observe([%.4g, %v...]): "+
				"got %.4g, want %.4g",
				c.lambda, c.y, llo, ll)
		}
		if len(c.y) == 1 {
			ll1 := Expon.Logp(c.lambda, c.y[0])
			if math.Abs(ll-ll1) > 1E-6 {
				t.Errorf("Wrong result of Logp(%.4g, %.4g): "+
					"got %.4g, want %.4g",
					c.lambda, c.y[0], ll1, ll)
			}
		}
	}
}

func TestGamma(t *testing.T) {
	for _, c := range []struct {
		alpha, beta float64
		y           []float64
		ll          float64
	}{
		{1., 1., []float64{1.}, -1},
		{2., 2., []float64{2.}, -1.9205584583201638},
		{1., 1., []float64{2., 3.}, -5},
	} {
		ll := Gamma.Logps(c.alpha, c.beta, c.y...)
		if math.Abs(ll-c.ll) > 1E-6 {
			t.Errorf("Wrong logpdf of Gamma(%.v|%.v,%.v): "+
				"got %.4g, want %.4g",
				c.y, c.alpha, c.beta, ll, c.ll)
		}
		llo := Gamma.Observe(append([]float64{c.alpha, c.beta}, c.y...))
		if math.Abs(ll-llo) > 1E-6 {
			t.Errorf("Wrong result of Observe([%.4g, %.4g, %v...]): "+
				"got %.4g, want %.4g",
				c.alpha, c.beta, c.y, llo, ll)
		}
		if len(c.y) == 1 {
			ll1 := Gamma.Logp(c.alpha, c.beta, c.y[0])
			if math.Abs(ll-ll1) > 1E-6 {
				t.Errorf("Wrong result of Logp(%.4g, %.4g, %.4g): "+
					"got %.4g, want %.4g",
					c.beta, c.alpha, c.y[0], ll1, ll)
			}
		}
	}
}

func TestBeta(t *testing.T) {
	for _, c := range []struct {
		alpha, beta float64
		y           []float64
		ll          float64
	}{
		{1., 1., []float64{0.5}, 0},
		{2., 3., []float64{.25}, 0.523248143764548},
		{3., 1., []float64{0.3, 0.6}, -1.2323722788476341},
	} {
		ll := Beta.Logps(c.alpha, c.beta, c.y...)
		if math.Abs(ll-c.ll) > 1E-6 {
			t.Errorf("Wrong logpdf of Beta(%.v|%.v,%.v): "+
				"got %.4g, want %.4g",
				c.y, c.alpha, c.beta, ll, c.ll)
		}
		llo := Beta.Observe(append([]float64{c.alpha, c.beta}, c.y...))
		if math.Abs(ll-llo) > 1E-6 {
			t.Errorf("Wrong result of Observe([%.4g, %.4g, %v...]): "+
				"got %.4g, want %.4g",
				c.alpha, c.beta, c.y, llo, ll)
		}
		if len(c.y) == 1 {
			ll1 := Beta.Logp(c.alpha, c.beta, c.y[0])
			if math.Abs(ll-ll1) > 1E-6 {
				t.Errorf("Wrong result of Logp(%.4g, %.4g, %.4g): "+
					"got %.4g, want %.4g",
					c.alpha, c.beta, c.y[0], ll1, ll)
			}
		}
	}
}

func TestDirichlet(t *testing.T) {
	for _, c := range []struct {
		n     int
		alpha []float64
		y     [][]float64
		ll    float64
	}{
		{
			2,
			[]float64{1., 1.},
			[][]float64{{0.75, 0.25}},
			0,
		},
		{
			3,
			[]float64{0.5, 1., 2.},
			[][]float64{{0.1, 0.1, 0.8}},
			1.5567576546051873,
		},
		{
			2,
			[]float64{3., 1.},
			[][]float64{
				{0.4, 0.6},
				{0.6, 0.4},
			},
			-0.6570081339440723,
		},
	} {
		dist := Dirichlet{c.n}
		ll := dist.Logps(c.alpha, c.y...)
		if math.Abs(ll-c.ll) > 1E-6 {
			t.Errorf("Wrong logpdf of Dirichlet(%v|%v): "+
				"got %.4g, want %.4g",
				c.y, c.alpha, ll, c.ll)
		}
		x := c.alpha
		for _, y := range c.y {
			x = append(x, y...)
		}
		llo := dist.Observe(x)
		if math.Abs(ll-llo) > 1E-6 {
			t.Errorf("Wrong result of Observe(%v..., %v...): "+
				"got %.4g, want %.4g",
				c.alpha, c.y, llo, ll)
		}
		if len(c.y) == 1 {
			ll1 := dist.Logp(c.alpha, c.y[0])
			if math.Abs(ll-ll1) > 1E-6 {
				t.Errorf("Wrong result of Logp(%v, %v): "+
					"got %.4g, want %.4g",
					c.alpha, c.y[0], ll1, ll)
			}
		}
	}
}

func TestSoftMax(t *testing.T) {
	for _, c := range []struct {
		x []float64
		p []float64
	}{
		{
			[]float64{0., 0.},
			[]float64{0.5, 0.5},
		},
		{
			[]float64{math.Log(1), math.Log(3), math.Log(6)},
			[]float64{0.1, 0.3, 0.6},
		},
	} {
		p := make([]float64, len(c.x))
		D.SoftMax(c.x, p)
		for i := range p {
			if math.Abs(p[i]-c.p[i]) > 1E-6 {
				t.Errorf("Wrong result of SoftMax(%v): "+
					"got %v, want %v", c.x, p, c.p)
				break
			}
		}
	}
}

func TestCategorical(t *testing.T) {
	for _, c := range []struct {
		n     int
		alpha []float64
		y     []float64
		ll    float64
	}{
		{
			2,
			[]float64{1., 1.},
			[]float64{0},
			-0.6931471805599453,
		},
		{
			3,
			[]float64{0.5, 2.5, 2.},
			[]float64{1},
			-0.6931471805599453,
		},
		{
			2,
			[]float64{3., 1.},
			[]float64{0, 1},
			-1.6739764335716716,
		},
	} {
		dist := Categorical{c.n}
		ll := dist.Logps(c.alpha, c.y...)
		if math.Abs(ll-c.ll) > 1E-6 {
			t.Errorf("Wrong logpdf of Categorical(%v|%v): "+
				"got %.4g, want %.4g",
				c.y, c.alpha, ll, c.ll)
		}
		x := append(c.alpha, c.y...)
		llo := dist.Observe(x)
		if math.Abs(ll-llo) > 1E-6 {
			t.Errorf("Wrong result of Observe(%v..., %v...): "+
				"got %.4g, want %.4g",
				c.alpha, c.y, llo, ll)
		}
		if len(c.y) == 1 {
			ll1 := dist.Logp(c.alpha, c.y[0])
			if math.Abs(ll-ll1) > 1E-6 {
				t.Errorf("Wrong result of Logp(%v, %v): "+
					"got %.4g, want %.4g",
					c.alpha, c.y[0], ll1, ll)
			}
		}
	}
}


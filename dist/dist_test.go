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
		lp        float64
	}{
		{0., 1., []float64{0.}, -0.9189385332046727},
		{1., 2., []float64{2.}, -1.737085713764618},
		{0., 1., []float64{-1., 0.}, -2.3378770664093453},
	} {
		lp := Normal.Logps(c.mu, c.sigma, c.y...)
		if math.Abs(lp-c.lp) > 1e-6 {
			t.Errorf("Wrong logpdf of Logps(%.v|%.v, %.v): "+
				"got %.4g, want %.4g",
				c.y, c.mu, c.sigma, lp, c.lp)
		}
		lpo := Normal.Observe(append([]float64{c.mu, c.sigma}, c.y...))
		if math.Abs(lp-lpo) > 1e-6 {
			t.Errorf("Wrong result of Observe([%.4g, %.4g, %v...]): "+
				"got %.4g, want %.4g",
				c.mu, c.sigma, c.y, lpo, lp)
		}
		if len(c.y) == 1 {
			lp1 := Normal.Logp(c.mu, c.sigma, c.y[0])
			if math.Abs(lp-lp1) > 1e-6 {
				t.Errorf("Wrong result of Logp(%.4g, %.4g, %.4g): "+
					"got %.4g, want %.4g",
					c.mu, c.sigma, c.y[0], lp1, lp)
			}
		}
	}
}

func TestCauchy(t *testing.T) {
	for _, c := range []struct {
		x0, gamma float64
		y         []float64
		lp        float64
	}{
		{0., 1., []float64{0.}, -1.1447298858494002},
		{1., 2.5, []float64{2.}, -2.2094406228418286},
		{0., 5, []float64{-1., 0.}, -5.5475563097202825},
	} {
		lp := Cauchy.Logps(c.x0, c.gamma, c.y...)
		if math.Abs(lp-c.lp) > 1e-6 {
			t.Errorf("Wrong logpdf of Logps(%.v|%.v, %.v): "+
				"got %.4g, want %.4g",
				c.y, c.x0, c.gamma, lp, c.lp)
		}
		lpo := Cauchy.Observe(append([]float64{c.x0, c.gamma}, c.y...))
		if math.Abs(lp-lpo) > 1e-6 {
			t.Errorf("Wrong result of Observe([%.4g, %.4g, %v...]): "+
				"got %.4g, want %.4g",
				c.x0, c.gamma, c.y, lpo, lp)
		}
		if len(c.y) == 1 {
			lp1 := Cauchy.Logp(c.x0, c.gamma, c.y[0])
			if math.Abs(lp-lp1) > 1e-6 {
				t.Errorf("Wrong result of Logp(%.4g, %.4g, %.4g): "+
					"got %.4g, want %.4g",
					c.x0, c.gamma, c.y[0], lp1, lp)
			}
		}
	}
}

func TestExpon(t *testing.T) {
	for _, c := range []struct {
		lambda float64
		y      []float64
		lp     float64
	}{
		{1., []float64{1.}, -1},
		{2., []float64{2.}, -3.3068528194400546},
		{1., []float64{1., 2.}, -3},
	} {
		lp := Expon.Logps(c.lambda, c.y...)
		if math.Abs(lp-c.lp) > 1e-6 {
			t.Errorf("Wrong logpdf of Logps(%.v|%.v): "+
				"got %.4g, want %.4g",
				c.y, c.lambda, lp, c.lp)
		}
		lpo := Expon.Observe(append([]float64{c.lambda}, c.y...))
		if math.Abs(lp-lpo) > 1e-6 {
			t.Errorf("Wrong result of Observe([%.4g, %v...]): "+
				"got %.4g, want %.4g",
				c.lambda, c.y, lpo, lp)
		}
		if len(c.y) == 1 {
			lp1 := Expon.Logp(c.lambda, c.y[0])
			if math.Abs(lp-lp1) > 1e-6 {
				t.Errorf("Wrong result of Logp(%.4g, %.4g): "+
					"got %.4g, want %.4g",
					c.lambda, c.y[0], lp1, lp)
			}
		}
	}
}

func TestGamma(t *testing.T) {
	for _, c := range []struct {
		alpha, beta float64
		y           []float64
		lp          float64
	}{
		{1., 1., []float64{1.}, -1},
		{2., 2., []float64{2.}, -1.9205584583201638},
		{1., 1., []float64{2., 3.}, -5},
	} {
		lp := Gamma.Logps(c.alpha, c.beta, c.y...)
		if math.Abs(lp-c.lp) > 1e-6 {
			t.Errorf("Wrong logpdf of Logps(%.v|%.v,%.v): "+
				"got %.4g, want %.4g",
				c.y, c.alpha, c.beta, lp, c.lp)
		}
		lpo := Gamma.Observe(append([]float64{c.alpha, c.beta}, c.y...))
		if math.Abs(lp-lpo) > 1e-6 {
			t.Errorf("Wrong result of Observe([%.4g, %.4g, %v...]): "+
				"got %.4g, want %.4g",
				c.alpha, c.beta, c.y, lpo, lp)
		}
		if len(c.y) == 1 {
			lp1 := Gamma.Logp(c.alpha, c.beta, c.y[0])
			if math.Abs(lp-lp1) > 1e-6 {
				t.Errorf("Wrong result of Logp(%.4g, %.4g, %.4g): "+
					"got %.4g, want %.4g",
					c.beta, c.alpha, c.y[0], lp1, lp)
			}
		}
	}
}

func TestBeta(t *testing.T) {
	for _, c := range []struct {
		alpha, beta float64
		y           []float64
		lp          float64
	}{
		{1., 1., []float64{0.5}, 0},
		{2., 3., []float64{.25}, 0.523248143764548},
		{3., 1., []float64{0.3, 0.6}, -1.2323722788476341},
	} {
		lp := Beta.Logps(c.alpha, c.beta, c.y...)
		if math.Abs(lp-c.lp) > 1e-6 {
			t.Errorf("Wrong logpdf of Logps(%.v|%.v,%.v): "+
				"got %.4g, want %.4g",
				c.y, c.alpha, c.beta, lp, c.lp)
		}
		lpo := Beta.Observe(append([]float64{c.alpha, c.beta}, c.y...))
		if math.Abs(lp-lpo) > 1e-6 {
			t.Errorf("Wrong result of Observe([%.4g, %.4g, %v...]): "+
				"got %.4g, want %.4g",
				c.alpha, c.beta, c.y, lpo, lp)
		}
		if len(c.y) == 1 {
			lp1 := Beta.Logp(c.alpha, c.beta, c.y[0])
			if math.Abs(lp-lp1) > 1e-6 {
				t.Errorf("Wrong result of Logp(%.4g, %.4g, %.4g): "+
					"got %.4g, want %.4g",
					c.alpha, c.beta, c.y[0], lp1, lp)
			}
		}
	}
}

func TestDirichlet(t *testing.T) {
	for _, c := range []struct {
		n     int
		alpha []float64
		y     [][]float64
		lp    float64
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
		lp := Dir.Logps(c.alpha, c.y...)
		if math.Abs(lp-c.lp) > 1e-6 {
			t.Errorf("Wrong logpdf of Logps(%v|%v): "+
				"got %.4g, want %.4g",
				c.y, c.alpha, lp, c.lp)
		}
		dist := Dirichlet{c.n}
		x := c.alpha
		for _, y := range c.y {
			x = append(x, y...)
		}
		lpo := dist.Observe(x)
		if math.Abs(lp-lpo) > 1e-6 {
			t.Errorf("Wrong result of Observe(%v..., %v...): "+
				"got %.4g, want %.4g",
				c.alpha, c.y, lpo, lp)
		}
		if len(c.y) == 1 {
			lp1 := Dir.Logp(c.alpha, c.y[0])
			if math.Abs(lp-lp1) > 1e-6 {
				t.Errorf("Wrong result of Logp(%v, %v): "+
					"got %.4g, want %.4g",
					c.alpha, c.y[0], lp1, lp)
			}
		}
	}
}

func TestBernoulli(t *testing.T) {
	for _, c := range []struct {
		p float64
		y []float64
		lp float64
	}{
		{
			0.5,
			[]float64{0},
			-0.6931471805599453,
		},
		{
			0.5,
			[]float64{1},
			-0.6931471805599453,
		},
		{
			0.75,
			[]float64{0, 1},
			-1.6739764335716716,
		},
	} {
		lp := Bernoulli.Logps(c.p, c.y...)
		if math.Abs(lp-c.lp) > 1e-6 {
			t.Errorf("Wrong logpdf of Logps(%v|%v): "+
				"got %.4g, want %.4g",
				c.y, c.p, lp, c.lp)
		}
		x := append([]float64{c.p}, c.y...)
		lpo := Bernoulli.Observe(x)
		if math.Abs(lp-lpo) > 1e-6 {
			t.Errorf("Wrong result of Observe(%v..., %v...): "+
				"got %.4g, want %.4g",
				c.p, c.y, lpo, lp)
		}
		if len(c.y) == 1 {
			lp1 := Bernoulli.Logp(c.p, c.y[0])
			if math.Abs(lp-lp1) > 1e-6 {
				t.Errorf("Wrong result of Logp(%v, %v): "+
					"got %.4g, want %.4g",
					c.p, c.y[0], lp1, lp)
			}
		}
	}
}


func TestCategorical(t *testing.T) {
	for _, c := range []struct {
		n     int
		alpha []float64
		y     []float64
		lp    float64
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
		lp := Cat.Logps(c.alpha, c.y...)
		if math.Abs(lp-c.lp) > 1e-6 {
			t.Errorf("Wrong logpdf of Logps(%v|%v): "+
				"got %.4g, want %.4g",
				c.y, c.alpha, lp, c.lp)
		}
		dist := Categorical{c.n}
		x := append(c.alpha, c.y...)
		lpo := dist.Observe(x)
		if math.Abs(lp-lpo) > 1e-6 {
			t.Errorf("Wrong result of Observe(%v..., %v...): "+
				"got %.4g, want %.4g",
				c.alpha, c.y, lpo, lp)
		}
		if len(c.y) == 1 {
			lp1 := Cat.Logp(c.alpha, c.y[0])
			if math.Abs(lp-lp1) > 1e-6 {
				t.Errorf("Wrong result of Logp(%v, %v): "+
					"got %.4g, want %.4g",
					c.alpha, c.y[0], lp1, lp)
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
			if math.Abs(p[i]-c.p[i]) > 1e-6 {
				t.Errorf("Wrong result of SoftMax(%v): "+
					"got %v, want %v", c.x, p, c.p)
				break
			}
		}
	}
}

func TestLogSumExp(t *testing.T) {
	for _, c := range []struct {
		x []float64
		y float64
	}{
		{[]float64{0, 0}, 0.693147181},
		{[]float64{-1, -1}, -0.306852819},
		{[]float64{0, 1}, 1.313261687},
		{[]float64{1, 0, -1}, 1.407605964},
	} {
		y := D.LogSumExp(c.x)
		if math.Abs(y-c.y) > 1e-6 {
			t.Errorf("Wrong LogSumExp(%v): "+
				"got %.4g, want %.4g", c.x, y, c.y)
		}
	}
}

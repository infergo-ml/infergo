package infer

// Markov Chain Monte Carlo for approximation of the posterior.

import (
	"bitbucket.org/dtolpin/infergo/ad"
	"bitbucket.org/dtolpin/infergo/model"
	"math"
	"math/rand"
	"time"
)

// MCMC is the interface of MCMC samplers.
type MCMC interface {
	Sample(
		m model.Model,
		x []float64,
		samples chan []float64,
	)
	Stop()
}

// sampler is the structure for embedding into concrete samplers.
type sampler struct {
	stop    bool
	samples chan []float64
}

// Helper functions

// Stop stops a sampler gracefully, using the samples channel
// for synchronization. A part of the MCMC interface.
// samplers.
func (s *sampler) Stop() {
	s.stop = true
	for {
		select {
		case _, ok := <-s.samples:
			if !ok { // channel closed, safe to leave
				return
			}
		default:
			time.Sleep(1)
		}
	}
}

// energy computes the energy of a particle; used
// by HMC variants.
func energy(l float64, r []float64) float64 {
	k := 0.
	for i := 0; i != len(r); i++ {
		k += r[i] * r[i]
	}
	return l + 0.5*k
}

// leapfrog advances x and r a single 'leapfrog'; used
// by HMC variants.
func leapfrog(
	m model.Model,
	gradp *[]float64,
	x []float64,
	r []float64,
	eps float64,
) (l float64) {
	for i := 0; i != len(x); i++ {
		r[i] += 0.5 * eps * (*gradp)[i]
		x[i] += eps * r[i]
	}
	l, *gradp = m.Observe(x), ad.Gradient()
	for i := 0; i != len(x); i++ {
		r[i] += 0.5 * eps * (*gradp)[i]
	}
	return l
}

// Vanilla Hamiltonian Monte Carlo Sampler.
type HMC struct {
	sampler
	L   int     // number of leapfrog steps
	Eps float64 // leapfrog step size
}

func (hmc *HMC) Sample(
	m model.Model,
	x []float64,
	samples chan []float64,
) {
	hmc.samples = samples // Stop needs access to samples
	go func() {
		backup := make([]float64, len(x))
		for iter := 0; ; iter++ {
			if hmc.stop {
				close(samples)
				break
			}
			// sample the next r
			r := make([]float64, len(x))
			for i := 0; i != len(x); i++ {
				r[i] = rand.NormFloat64()
			}

			// Back up the current value of x
			copy(backup, x)

			l0, grad := m.Observe(x), ad.Gradient()
			e0 := energy(l0, r) // initial energy
			var l float64
			for i := 0; i != hmc.L; i++ {
				l = leapfrog(m, &grad, x, r, hmc.Eps)
			}
			e := energy(l, r) // final energy

			// Accept with MH probability.
			if e-e0 < math.Log(1.-rand.Float64()) {
				// Rejected, restore x from backup.
				copy(x, backup)
			}

			// write a sample to the channel
			sample := make([]float64, len(x))
			copy(sample, x)
			samples <- sample
		}
	}()
}

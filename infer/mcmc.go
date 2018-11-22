package infer

// Markov Chain Monte Carlo for approximation of the posterior.

import (
	"bitbucket.org/dtolpin/infergo/ad"
	"bitbucket.org/dtolpin/infergo/model"
	"log"
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

// sampler is the structure for embedding into concrete
// samplers.
type sampler struct {
	stop    bool
	samples chan []float64
	// Statistics
	NAcc, NRej int // the number of accepted and rejected samples
}

// Helper functions

// Stop stops a sampler gracefully, using the samples channel
// for synchronization. Stop must be called before further calls
// to differentiated code. A part of the MCMC interface.
func (s *sampler) Stop() {
	s.stop = true
	// The differentiated code is not thread-safe, hence
	// we must exhaust samples before returning from Stop,
	// so that an Observe called afterwards does not overlap
	// with an Observe called in the sampler.
	for {
		select {
		case _, ok := <-s.samples:
			if !ok { // channel closed, safe to leave
				return
			}
			// Discard the sample and continue.
		default:
			// No data but the channel is open, the sampler is
			// computing a sample. Wait for the computation to
			// complete.
			time.Sleep(1000) // one millisecond
		}
	}
}

// energy computes the energy of a particle; used
// by HMC variants.
func energy(l float64, r []float64) float64 {
	k := 0.
	for _, ri := range r {
		k += ri * ri
	}
	return l + 0.5*k
}

// leapfrog advances x and r a single 'leapfrog'; used
// by HMC variants.
func leapfrog(
	m model.Model,
	gradp *[]float64,
	x, r []float64,
	eps float64,
) (l float64) {
	for i := range x {
		r[i] += 0.5 * eps * (*gradp)[i]
		x[i] += eps * r[i]
	}
	l, *gradp = m.Observe(x), ad.Gradient()
	if math.IsNaN(l) {
		panic("energy diverged")
	}
	for i := range x {
		r[i] += 0.5 * eps * (*gradp)[i]
	}
	return l
}

// Vanilla Hamiltonian Monte Carlo Sampler.
type HMC struct {
	sampler
	// Parameters
	L   int     // number of leapfrog steps
	Eps float64 // leapfrog step size
}

func (hmc *HMC) Sample(
	m model.Model,
	x []float64,
	samples chan []float64,
) {
	hmc.samples = samples // Stop needs access to samples
	hmc.setDefaults()
	go func() {
		// Close samples on exit
		defer close(samples)
		// Intercept errors deep inside the algorithm
		// and report them.
		defer func() {
			if r := recover(); r != nil {
				log.Printf("ERROR: HMC: %v", r)
			}
		}()
		backup := make([]float64, len(x))
		r := make([]float64, len(x))
		for {
			if hmc.stop {
				break
			}
			// Sample the next r.
			for i := range r {
				r[i] = rand.NormFloat64()
			}

			// Back up the current value of x.
			copy(backup, x)

			l0, grad := m.Observe(x), ad.Gradient()
			e0 := energy(l0, r) // initial energy
			var l float64
			for i := 0; i != hmc.L; i++ {
				l = leapfrog(m, &grad, x, r, hmc.Eps)
			}
			e := energy(l, r) // final energy

			// Accept with MH probability.
			if e-e0 >= math.Log(1.-rand.Float64()) {
				hmc.NAcc++
			} else {
				// Rejected, restore x from backup.
				copy(x, backup)
				hmc.NRej++
			}

			// Write a sample to the channel.
			sample := make([]float64, len(x))
			copy(sample, x)
			samples <- sample
		}
	}()
}

// setDefaults sets the default value for auxiliary parameters
// of HMC.
func (hmc *HMC) setDefaults() {
    if hmc.L == 0 {
        hmc.L = 10
    }
}

// No U-Turn Sampler (https://arxiv.org/abs/1111.4246).
type NUTS struct {
	sampler
	// Parameters
	Eps   float64 // step size
	Delta float64 // lower bound on energy for doubling
	// Statistics
	Depth [][2]float64 // depth belief
	// Depth belief is encoded as a vector of beta-bernoulli
	// distributions. If the depth is greater than the element's
	// index i, Depth[i][0] is incremented; for index depth,
	// Depth[depth][1] is incremented.
}

func (nuts *NUTS) Sample(
	m model.Model,
	x []float64,
	samples chan []float64,
) {
	nuts.samples = samples // Stop needs access to samples
	nuts.setDefaults()
	go func() {
		// Close samples on exit
		defer close(samples)
		// Intercept errors deep inside the algorithm
		// and report them.
		defer func() {
			if r := recover(); r != nil {
				log.Printf("ERROR: NUTS: %v", r)
			}
		}()

		backup := make([]float64, len(x))
		r := make([]float64, len(x))
		for {
			if nuts.stop {
				break
			}
			// Sample the next r.
			for i := range r {
				r[i] = rand.NormFloat64()
			}

			// Back up the current value of x.
			copy(backup, x)

			// Compute the energy
			l, grad := m.Observe(x), ad.Gradient()
			e := energy(l, r)

			// Sample the slice variable
			logu := math.Log((1. - rand.Float64())) + e

			// Initialize the tree
			xl, rl, xr, rr, depth, nelem := x, r, x, r, 0, 1.
			// Integrate forward
			accepted := false
			for {
				// Build left or right subtree
				var (
					nelemSub float64
					stop     bool
				)
				if rand.Float64() < 0.5 { // choose direction
					dir := -1.
					xl, rl, _, _, x, nelemSub, stop =
						nuts.buildTree(m, &grad,
							x, r, logu, dir, depth)
				} else {
					dir := 1.
					_, _, xr, rr, x, nelemSub, stop =
						nuts.buildTree(m, &grad,
							x, r, logu, dir, depth)
				}

				// Accept or reject
				if nelemSub/nelem > rand.Float64() {
					accepted = true
				} else {
					// Rejected, restore x from backup.
					copy(x, backup)
				}

				if stop || uTurn(xl, rl, xr, rr) {
					break
				}

				nelem += nelemSub
				depth++
			}

			// Collect statistics
			if accepted {
				nuts.NAcc++
			} else {
				nuts.NRej++
			}
			nuts.updateDepth(depth)

			// Write a sample to the channel.
			sample := make([]float64, len(x))
			copy(sample, x)
			samples <- sample
		}
	}()
}

func (nuts *NUTS) buildTree(
	m model.Model,
	gradp *[]float64,
	x, r []float64,
	logu float64,
	dir float64,
	depth int,
) (
	xl, rl, xr, rr, x_ []float64,
	nelem float64,
	stop bool,
) {
	if depth == 0 {
		l := leapfrog(m, gradp, x, r, dir*nuts.Eps)
		if energy(l, r) >= logu {
			nelem = 1.
		}
		if energy(l, r)+nuts.Delta <= logu {
			stop = true
		}
		return x, r, x, r, x, nelem, stop
	} else {
		xl, rl, xr, rr, x, nelem, stop =
			nuts.buildTree(m, gradp, x, r, logu, dir, depth-1)
		if !stop {
			// We build a subtree and need a separate memory
			// for x and r.
			xSub := make([]float64, len(x))
			rSub := make([]float64, len(r))
			var nelemSub float64
			if dir == -1. {
				copy(xSub, xl)
				copy(rSub, rl)
				xl, rl, _, _, xSub, nelemSub, stop =
					nuts.buildTree(m, gradp,
						xSub, rSub, logu, dir, depth-1)
			} else {
				copy(xSub, xr)
				copy(rSub, rr)
				_, _, xr, rr, xSub, nelemSub, stop =
					nuts.buildTree(m, gradp,
						xSub, rSub, logu, dir, depth-1)
			}
			nelem += nelemSub
			stop = stop || uTurn(xl, rl, xr, rr)

			// Select uniformly from nodes.
			if nelemSub/nelem > rand.Float64() {
				x = xSub
			}
		}
		return xl, rl, xr, rr, x, nelem, stop
	}
}

// setDefaults sets the default value for auxiliary parameters
// of NUTS.
func (nuts *NUTS) setDefaults() {
	if nuts.Delta == 0 {
		nuts.Delta = 1E3
	}
}

// updateDepth updates depth beliefs.
func (nuts *NUTS) updateDepth(depth int) {
	if len(nuts.Depth) <= depth {
		nuts.Depth = append(nuts.Depth,
			make([][2]float64, depth-len(nuts.Depth)+1)...)
	}
	for i := 0; i != depth; i++ {
		nuts.Depth[i][0]++
	}
	nuts.Depth[depth][1]++
}

// MeanDepth returns the average observed depth.
func (nuts *NUTS) MeanDepth() float64 {
	meanDepth := 0.
	p := 1.
	for _, depth := range nuts.Depth {
		alpha, beta := depth[0], depth[1]
		p *= alpha / (alpha + beta)
		meanDepth += p
	}
	return meanDepth
}

// uTurn returns true iff there is a u-turn.
func uTurn(xl, rl, xr, rr []float64) bool {
	// Dot products of changes and moments to
	// stop on U-turns.
	dxrl, dxrr := 0., 0.
	for i := range xl {
		dxrl += (xr[i] - xl[i]) * rl[i]
		dxrr += (xr[i] - xl[i]) * rr[i]
	}
	return dxrl < 0 || dxrr < 0
}

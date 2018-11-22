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
		x_ := make([]float64, len(x))
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
			copy(x_, x)

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
				// Rejected, restore x.
				copy(x, x_)
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

		r := make([]float64, len(x))
		for {
			if nuts.stop {
				break
			}
			// Sample the next r.
			for i := range r {
				r[i] = rand.NormFloat64()
			}

			// Compute the energy.
			l, grad := m.Observe(x), ad.Gradient()
			e := energy(l, r)

			// Sample the slice variable
			logu := math.Log((1. - rand.Float64())) + e

			// Initialize the tree
			xl, rl, xr, rr, depth, nelem := x, r, x, r, 0, 1.
			// Integrate forward
			accepted := false
			for {
				// Build left or right subtree.

				// Choose direction.
				var dir float64
				if rand.Float64() < 0.5 {
					dir = -1.
				} else {
					dir = 1.
				}

				var (
					x_     []float64
					nelem_ float64
					stop   bool
				)
				xl, rl, xr, rr, x_, nelem_, stop =
					nuts.buildLeftOrRightTree(m, &grad,
						xl, rl, xr, rr, logu, dir, depth)

				// Accept or reject
				if nelem_/nelem > rand.Float64() {
					accepted = true
					x = x_
					copy(x, x_)
				}

				if stop {
					break
				}

				nelem += nelem_
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

func (nuts *NUTS) buildLeftOrRightTree(
	m model.Model,
	gradp *[]float64,
	xl, rl, xr, rr []float64,
	logu float64,
	dir float64,
	depth int,
) (
	_, _, _, _, x []float64,
	nelem float64,
	stop bool,
) {
	// We are building a subtree and need memory for each new
	// node.
	x_ := make([]float64, len(xl))
	r_ := make([]float64, len(rl))
	if dir == -1. {
		copy(x_, xl)
		copy(r_, rl)
		xl, rl, _, _, x, nelem, stop = nuts.buildTree(m, gradp,
			x_, r_, logu, dir, depth)
	} else {
		copy(x_, xr)
		copy(r_, rr)
		_, _, xr, rr, x, nelem, stop = nuts.buildTree(m, gradp,
			x_, r_, logu, dir, depth)
	}

	if uTurn(xl, xr, rl) || uTurn(xl, xr, rr) {
		stop = true
	}

	return xl, rl, xr, rr, x, nelem, stop
}

func (nuts *NUTS) buildTree(
	m model.Model,
	gradp *[]float64,
	x, r []float64,
	logu float64,
	dir float64,
	depth int,
) (
	xl, rl, xr, rr, _ []float64,
	nelem float64,
	stop bool,
) {
	if depth == 0 {
		// Base case: single leapfrog
		l := leapfrog(m, gradp, x, r, dir*nuts.Eps)
		if energy(l, r) >= logu {
			nelem = 1.
		}
		if energy(l, r)+nuts.Delta <= logu {
			stop = true
		}
		return x, r, x, r, x, nelem, stop
	} else {
		depth--

		xl, rl, xr, rr, x, nelem, stop =
			nuts.buildTree(m, gradp, x, r, logu, dir, depth)
		if stop {
			return xl, rl, xr, rr, x, nelem, stop
		}

		xl, rl, xr, rr, x_, nelem_, stop :=
			nuts.buildLeftOrRightTree(m, gradp,
				xl, rl, xr, rr, logu, dir, depth)
		nelem += nelem_

		// Select uniformly from nodes.
		if nelem_/nelem > rand.Float64() {
			x = x_
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
func uTurn(xl, xr, r []float64) bool {
	// Dot product of changes and moment to
	// stop on U-turn.
	dxr := 0.
	for i := range xl {
		dxr += (xr[i] - xl[i]) * r[i]
	}
	return dxr < 0
}

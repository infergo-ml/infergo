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

// Sampler is the structure for embedding into concrete
// samplers.
type Sampler struct {
	Stopped bool
	Samples chan []float64
	// Statistics
	NAcc, NRej int // the number of accepted and rejected samples
}

// Helper functions

// Stop stops a sampler gracefully, using the samples channel
// for synchronization. Stop must be called before further calls
// to differentiated code. A part of the MCMC interface.
func (s *Sampler) Stop() {
	s.Stopped = true
	// The differentiated code is not necessarily thread-safe,
	// hence we must exhaust samples before returning from Stop,
	// so that an Observe called afterwards does not overlap
	// with an Observe called in the sampler.
	for {
		select {
		case _, ok := <-s.Samples:
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
	return l - 0.5*k
}

// leapfrog advances x and r a single 'leapfrog'; used
// by HMC variants.
func leapfrog(
	m model.Model,
	grad []float64,
	x, r []float64,
	eps float64,
) (l float64, _ []float64) {
	for i := range x {
		r[i] += 0.5 * eps * grad[i]
		x[i] += eps * r[i]
	}
	l, grad = m.Observe(x), model.Gradient(m)
	if math.IsNaN(l) {
		panic("energy diverged")
	}
	for i := range x {
		r[i] += 0.5 * eps * grad[i]
	}

	return l, grad
}

// clone clones state or momentum slice; used as poor man's
// copy-on-write.
func clone(x []float64) []float64 {
	x_ := make([]float64, len(x))
	copy(x_, x)
	return x_
}

// Vanilla Hamiltonian Monte Carlo Sampler.
type HMC struct {
	Sampler
	// Parameters
	L   int     // number of leapfrog steps
	Eps float64 // leapfrog step size
}

func (hmc *HMC) Sample(
	m model.Model,
	x []float64,
	samples chan []float64,
) {
	hmc.setDefaults()
	hmc.Samples = samples // Stop needs access to samples
	go func() {
		// On exit:
		// * drop the tape;
		defer ad.DropTape()
		// * close samples;
		defer close(samples)
		// * intercept errors deep inside the algorithm
		// and report them.
		defer func() {
			if r := recover(); r != nil {
				log.Printf("ERROR: HMC: %v", r)
			}
		}()
		r := make([]float64, len(x))
		x_ := make([]float64, len(x))
		for {
			if hmc.Stopped {
				break
			}
			// Sample the next r.
			for i := range r {
				r[i] = rand.NormFloat64()
			}

			l0, grad := m.Observe(x), model.Gradient(m)
			e0 := energy(l0, r) // initial energy
			var l float64
			copy(x_, x)
			for i := 0; i != hmc.L; i++ {
				l, grad = leapfrog(m, grad, x, r, hmc.Eps)
			}
			e := energy(l, r) // final energy

			// Accept with MH probability.
			if e-e0 >= math.Log(1-rand.Float64()) {
				hmc.NAcc++
			} else {
				// Rejected, restore x.
				x, x_ = x_, x
				hmc.NRej++
			}

			// Write a sample to the channel.
			// x is modified in place by leapfrog and
			// therefore must be cloned.
			samples <- clone(x)
		}
	}()
}

// setDefaults sets the default value for auxiliary parameters.
func (hmc *HMC) setDefaults() {
	if hmc.L == 0 {
		hmc.L = 10
	}
}

// No U-Turn Sampler (https://arxiv.org/abs/1111.4246).
type NUTS struct {
	Sampler
	// Parameters
	Eps      float64 // step size
	Delta    float64 // lower bound on energy for doubling
	MaxDepth int     // maximum depth
	// Statistics
	// Depth belief is encoded as a vector of beta-bernoulli
	// distributions. If the depth is greater than the element's
	// index i, Depth[i][0] is incremented; for index depth,
	// Depth[depth][1] is incremented.
	Depth [][2]float64 // depth belief
	// Cached model run
	x    []float64
	l    float64
	grad []float64
}

func (nuts *NUTS) Sample(
	m model.Model,
	x []float64,
	samples chan []float64,
) {
	nuts.setDefaults()
	nuts.Samples = samples // Stop needs access to samples
	nuts.x = nil           // invalidate gradient cache
	go func() {
		// On exit:
		// * drop the tape;
		defer ad.DropTape()
		// * close samples;
		defer close(samples)
		// * intercept errors deep inside the algorithm
		// and report them.
		defer func() {
			if r := recover(); r != nil {
				log.Printf("ERROR: NUTS: %v", r)
			}
		}()

		r := make([]float64, len(x))
		for {
			if nuts.Stopped {
				break
			}

			// Sample the next r.
			for i := range r {
				r[i] = rand.NormFloat64()
			}

			// Compute the energy.
			l, _ := nuts.observe(m, x)
			e := energy(l, r)

			// Sample the slice variable
			logu := math.Log((1 - rand.Float64())) + e

			// Initialize the tree
			xl, rl, xr, rr, depth, nelem := x, r, x, r, 0, 1.
			// Integrate forward
			accepted := false
			for {
				// Build left or right subtree.

				// Choose direction.
				var dir float64
				if rand.Float64() < 0.5 {
					dir = -1
				} else {
					dir = 1
				}

				var (
					x_     []float64
					nelem_ float64
					stop   bool
				)
				xl, rl, xr, rr, x_, nelem_, stop =
					nuts.buildLeftOrRightTree(
						m, xl, rl, xr, rr, logu, dir, depth)
				if stop {
					break
				}

				// Accept or reject
				if nelem_/nelem > rand.Float64() {
					accepted = true
					x = x_
				}

				nelem += nelem_
				depth++
				// Maximum depth of 0 (which is the default)
				// means unlimited depth.
				if depth == nuts.MaxDepth {
					break
				}
			}

			// Collect statistics
			if accepted {
				nuts.NAcc++
			} else {
				nuts.NRej++
			}
			nuts.updateDepth(depth)

			// Write a sample to the channel.
			// x need not be cloned here since it is cloned
			// before the call to leapfrog.
			samples <- x
		}
	}()
}

func (nuts *NUTS) buildLeftOrRightTree(
	m model.Model,
	xl, rl, xr, rr []float64,
	logu float64,
	dir float64,
	depth int,
) (
	_, _, _, _, x []float64,
	nelem float64,
	stop bool,
) {
	if dir == -1 {
		xl, rl, _, _, x, nelem, stop =
			nuts.buildTree(m, xl, rl, logu, dir, depth)
	} else {
		_, _, xr, rr, x, nelem, stop =
			nuts.buildTree(m, xr, rr, logu, dir, depth)
	}

	if uTurn(xl, xr, rl) || uTurn(xl, xr, rr) {
		stop = true
	}

	return xl, rl, xr, rr, x, nelem, stop
}

func (nuts *NUTS) buildTree(
	m model.Model,
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
		// Base case: single leapfrog. State x and momentum r
		// are copied because leapfrog modifies them in place.
		x, r := clone(x), clone(r)
		_, grad := nuts.observe(m, x)
		l, grad := leapfrog(m, grad, x, r, dir*nuts.Eps)
		// Cache model run inside leapfrog
		nuts.x, nuts.l, nuts.grad = x, l, grad
		if energy(l, r) >= logu {
			nelem = 1
		}
		if energy(l, r)+nuts.Delta <= logu {
			stop = true
		}
		return x, r, x, r, x, nelem, stop
	} else {
		depth--

		xl, rl, xr, rr, x, nelem, stop =
			nuts.buildTree(m, x, r, logu, dir, depth)
		if stop {
			return xl, rl, xr, rr, x, nelem, stop
		}

		xl, rl, xr, rr, x_, nelem_, stop :=
			nuts.buildLeftOrRightTree(
				m, xl, rl, xr, rr, logu, dir, depth)
		nelem += nelem_

		// Select uniformly from nodes.
		if nelem_/nelem > rand.Float64() {
			x = x_
		}

		return xl, rl, xr, rr, x, nelem, stop
	}
}

// observe is a cached call to Observe and Gradient so that we
// re-run the model on change of direction, but re-use the earlier
// computed energy and gradient when possible.
func (nuts *NUTS) observe(m model.Model, x []float64) (
	/*l*/ float64,
	/*grad*/ []float64,
) {
	cached := nuts.x != nil
	if cached {
		for i := range x {
			if x[i] != nuts.x[i] {
				cached = false
				break
			}
		}
	}
	if !cached {
		nuts.x, nuts.l, nuts.grad = x,
			m.Observe(x), model.Gradient(m)
	}
	return nuts.l, nuts.grad
}

// setDefaults sets the default value for auxiliary parameters.
func (nuts *NUTS) setDefaults() {
	if nuts.Delta == 0 {
		nuts.Delta = 1e3
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
	dot := 0.
	for i := range xl {
		dot += (xr[i] - xl[i]) * r[i]
	}
	return dot < 0
}

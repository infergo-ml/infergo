package main

import (
	"bitbucket.org/dtolpin/infergo/ad"
	. "bitbucket.org/dtolpin/infergo/examples/ppv/model/ad"
	"bitbucket.org/dtolpin/infergo/infer"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"
)

// Command line arguments

var (
	OPT   = true
	POST  = true
	RATE  = 0.01
	DECAY = 0.998
	GAMMA = 0.9
	EPS   = 1E-6
	STEP  = 0.1
	NBURN = 0
	NITER = 100
	NADPT = 10
	DEPTH = 3.
)

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
	flag.Usage = func() {
		fmt.Printf(`Inferring best bandwidth. Usage:
		goppv [OPTIONS]` + "\n")
		flag.PrintDefaults()
	}
	flag.BoolVar(&OPT, "opt", OPT, "optimize")
	flag.BoolVar(&POST, "post", POST, "infer posterior")
	flag.Float64Var(&RATE, "rate", RATE, "learning rate")
	flag.Float64Var(&DECAY, "decay", DECAY, "rate decay")
	flag.Float64Var(&GAMMA, "gamma", GAMMA, "gradient momentum factor")
	flag.Float64Var(&EPS, "eps", EPS, "target accuracy")
	flag.Float64Var(&STEP, "step", STEP, "NUTS step")
	flag.IntVar(&NBURN, "nburn", NBURN, "number of burn-in iterations")
	flag.IntVar(&NITER, "niter", NITER, "number of iterations")
	flag.IntVar(&NADPT, "nadpt", NADPT, "number of steps per adaptation")
	flag.Float64Var(&DEPTH, "depth", DEPTH, "target NUTS tree depth")
	log.SetFlags(0)
}

func main() {
	flag.Parse()
	if NBURN == 0 {
		NBURN = NITER
	}

	if flag.NArg() > 1 {
		log.Printf("unexpected positional arguments: %v",
			flag.Args())
		os.Exit(1)
	}

	// Get the data
	var data []int
	if flag.NArg() == 1 {
		// Read the CSV
		fname := flag.Arg(0)
		file, err := os.Open(fname)
		if err != nil {
			log.Fatalf("Cannot open data file %q: %v", fname, err)
		}
		rdr := csv.NewReader(file)
		for {
			record, err := rdr.Read()
			if err == io.EOF {
				break
			}
			value, err := strconv.Atoi(record[0])
			if err != nil {
				log.Fatalf("invalid data: %v", err)
			}
			data = append(data, value)
		}
		file.Close()
	} else {
		// Use an embedded data set, for self-check
		data = []int{
			2, 10, 7, 3, 2, 4, 5, 1, 6, 4,
			4, 1, 1, 8, 3, 2, 1, 2, 1, 1,
			1, 3, 1, 3, 1, 2, 1, 1, 1, 2,
			3, 2, 1, 1, 5, 1, 7, 6, 2, 1,
			2, 1, 1, 1, 1, 1, 4, 1, 4, 1,
			4, 2, 3, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 3, 1, 1, 1, 1, 1, 1,
			5, 1, 1, 2, 2, 2, 1, 1, 2, 1,
			1, 1, 1, 2, 2, 1, 1, 1, 1, 1,
			2, 1, 1, 1, 1, 1, 1, 1, 2, 1,
		}
	}

	// Define the problem
	m := &Model{
		// Taken from stan worksheet
		PPV:            data,
		NPages:         10,
		PriorBandwidth: 100.,
	}
	// The parameter is log bandwidth
	x := make([]float64, 1)

	if OPT {
		// Set a starting point
		x[0] = math.Log(m.PriorBandwidth)
		// Compute log-likelihood of the starting point,
		// for comparison.
		ll0 := m.Observe(x)
		ad.Pop()

		// Run the optimizer
		opt := &infer.Momentum{
			Rate:  RATE,
			Decay: DECAY,
			Gamma: GAMMA,
		}
		var ll float64
		var iter int
		for iter = 0; iter != NITER; iter++ {
			xprev := x[0]
			ll, _ = opt.Step(m, x)
			if math.Abs(xprev-x[0]) < EPS*math.Abs(xprev+x[0]) {
				break
			}
		}

		log.Printf("Iterations: %d", iter)
		log.Printf("Best bandwidth: %.6g", math.Exp(x[0]))
		log.Printf("Log-likelihood: %.4g ⇒ %.4g", ll0, ll)

	}

	if POST {
		// Now let's infer the posterior with NUTS.
		nuts := &infer.NUTS{
			Eps: STEP,
		}
		samples := make(chan []float64)
		nuts.Sample(m, x, samples)
		// Adapt toward optimum tree depth.
		da := &infer.DepthAdapter{
			DualAveraging: infer.DualAveraging{Rate: RATE},
			Depth:         DEPTH,
			NAdpt:         NADPT,
		}
		da.Adapt(nuts, samples, NBURN)

		// Collect after burn-in
		n := 0.
		s := 0.
		s2 := 0.
		for i := 0; i != NITER; i++ {
			x := <-samples
			if len(x) == 0 {
				break
			}
			n++
			b := math.Exp(x[0])
			s += b
			s2 += b * b
		}
		nuts.Stop()
		mean := s / n
		stddev := math.Sqrt(s2/n - mean*mean)
		log.Printf("Bandwidth: mean=%.6g, stddev=%.6g", mean, stddev)
		log.Printf(`NUTS:
	accepted: %d
	rejected: %d
	rate: %.4g
	depth: %.4g`,
			nuts.NAcc, nuts.NRej,
			float64(nuts.NAcc)/float64(nuts.NAcc+nuts.NRej),
			nuts.MeanDepth())
	}
}

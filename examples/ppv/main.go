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
	"os"
	"strconv"
)

// Command line arguments

var (
	RATE  float64 = 0.01
	DECAY float64 = 0.998
	GAMMA float64 = 0.9
	NITER int     = 1000
	EPS   float64 = 1E-6
)

func init() {
	flag.Usage = func() {
		fmt.Printf(`Inferring best bandwidth. Usage:
		goppv [OPTIONS]` + "\n")
		flag.PrintDefaults()
	}
	flag.Float64Var(&RATE, "rate", RATE, "learning rate")
	flag.Float64Var(&DECAY, "decay", DECAY, "rate decay")
	flag.Float64Var(&GAMMA, "gamma", GAMMA, "gradient momentum factor")
	flag.IntVar(&NITER, "niter", NITER, "number of iterations")
	flag.Float64Var(&EPS, "eps", EPS, "target accuracy")
	log.SetFlags(0)
}

func main() {
	flag.Parse()

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

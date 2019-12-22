package main

import (
	"bitbucket.org/dtolpin/infergo/ad"
	. "bitbucket.org/dtolpin/infergo/examples/mt/model/ad"
	"bitbucket.org/dtolpin/infergo/infer"
	"encoding/csv"
	"flag"
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
	RATE   = 0.1
	DECAY  = 0.998
	GAMMA  = 0.9
	NITER  = 1000
	NSTEPS = 10
	STEP   = 0.5
	NGO    = 2
)

func init() {
	rand.Seed(time.Now().UnixNano())
	flag.Usage = func() {
		log.Printf(`Inferring parameters of the normal distribution:
		mt [OPTIONS] [data.csv]` + "\n")
		flag.PrintDefaults()
	}
	flag.Float64Var(&RATE, "rate", RATE, "learning rate")
	flag.Float64Var(&DECAY, "decay", DECAY, "rate decay")
	flag.Float64Var(&GAMMA, "gamma", GAMMA, "momentum factor")
	flag.IntVar(&NITER, "niter", NITER, "number of iterations")
	flag.IntVar(&NSTEPS, "nsteps", NSTEPS, "number of leapfrog steps")
	flag.Float64Var(&STEP, "step", STEP, "leapfrog step size")
	flag.IntVar(&NGO, "ngo", NGO, "number of inference goroutines")
	log.SetFlags(0)
	ad.MTSafeOn()
}

func main() {
	flag.Parse()

	if NGO > 1 && !ad.IsMTSafe() {
		log.Printf("Multithreading is not supported, " +
			"forcing -ngo 1.")
		NGO = 1
	}

	if flag.NArg() > 1 {
		log.Fatalf("unexpected positional arguments: %v",
			flag.Args()[1:])
	}

	// Get the data
	var data []float64
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
			value, err := strconv.ParseFloat(record[0], 64)
			if err != nil {
				log.Fatalf("invalid data: %v", err)
			}
			data = append(data, value)
		}
		file.Close()
	} else {
		// Use an embedded data set, for self-check
		data = []float64{
			-0.854, 1.067, -1.220, 0.818, -0.749,
			0.805, 1.443, 1.069, 1.426, 0.308}
	}
	m := &Model{Data: data}

	// Compute sample statistics, for comparison
	s := 0.
	s2 := 0.
	for i := 0; i != len(m.Data); i++ {
		x := m.Data[i]
		s += x
		s2 += x * x
	}
	sampleMean := s / float64(len(m.Data))
	sampleStddev := math.Sqrt(
		s2/float64(len(m.Data)) - sampleMean*sampleMean)

	// First estimate the maximum likelihood values.
	x := []float64{0.5 * rand.NormFloat64(), 1 + 0.5*rand.NormFloat64()}
	ll := m.Observe(x)
	printState := func(when string) {
		log.Printf(`%s:
	mean:   %.6g(≈%.6g)
	stddev: %.6g(≈%.6g)
	ll:     %.6g
`,
			when,
			x[0], sampleMean,
			math.Exp(x[1]), sampleStddev,
			ll)
	}

	printState("Initially")

	// Run the optimizer
	opt := &infer.Momentum{
		Rate:  RATE / float64(len(m.Data)),
		Decay: DECAY,
		Gamma: GAMMA,
	}
	for iter := 0; iter != NITER; iter++ {
		opt.Step(m, x)
	}

	ll = m.Observe(x)
	printState("Maximum likelihood")

	// Now let's infer the posterior with HMC.
	igos := make(chan int, NGO)
	finished := make(chan int, NGO)
	for igo := 0; igo != NGO; igo++ {
		igos <- igo
		go func() {
			igo := <-igos
			hmc := &infer.HMC{
				L:   NSTEPS,
				Eps: STEP / math.Sqrt(float64(len(m.Data))),
			}
			samples := make(chan []float64)
			hmc.Sample(m, x, samples)
			// Burn
			for i := 0; i != NITER; i++ {
				<-samples
			}

			// Collect after burn-in
			mean, stddev := 0., 0.
			n := 0.
			for i := 0; i != NITER; i++ {
				x := <-samples
				if len(x) == 0 {
					break
				}
				mean += x[0]
				stddev += math.Exp(x[1])
				n++
			}
			hmc.Stop()
			x[0], x[1] = mean/n, math.Log(stddev/n)
			ll = m.Observe(x)
			if NGO != 1 {
				log.Printf("\nGoroutine %v:", igo+1)
			}
			printState("Posterior")
			log.Printf(`HMC:
	accepted: %d
	rejected: %d
	rate: %.4g
`,
				hmc.NAcc, hmc.NRej,
				float64(hmc.NAcc)/float64(hmc.NAcc+hmc.NRej))
			finished <- igo
		}()
	}
	for jgo := 0; jgo != NGO; jgo++ {
		igo := <-finished
		if NGO != 1 {
			log.Printf("Goroutine %d finished.", igo+1)
		}
	}
}

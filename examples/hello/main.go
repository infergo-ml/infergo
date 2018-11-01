package main

import (
	. "bitbucket.org/dtolpin/infergo/examples/hello/model/ad"
	"bitbucket.org/dtolpin/infergo/infer"
	"encoding/csv"
	"flag"
	"io"
	"log"
	"math"
	"os"
	"strconv"
)

// Command line arguments

var (
	MEAN  float64 = 0.
	LOGV  float64 = 0.
	RATE  float64 = 0.01
	DECAY float64 = 0.995
	NITER int     = 100
)

func init() {
	flag.Usage = func() {
		log.Printf(`Inferring parameters of the normal distribution:
		hello [OPTIONS] [data.csv]` + "\n")
		flag.PrintDefaults()
	}
	flag.Float64Var(&MEAN, "mean", MEAN, "initial mean")
	flag.Float64Var(&LOGV, "logv", LOGV, "initial log var")
	flag.Float64Var(&RATE, "rate", RATE, "learning rate")
	flag.Float64Var(&DECAY, "decay", DECAY, "rate decay")
	flag.IntVar(&NITER, "niter", NITER, "number of iterations")
	log.SetFlags(0)
}

func main() {
	flag.Parse()

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
	sampleLogv := math.Log(
		s2/float64(len(m.Data)) - sampleMean*sampleMean)

	x := []float64{MEAN, LOGV}
	ll := m.Observe(x)
	printState := func(when string) {
		log.Printf(`
%s:
    mean: %.6g(≈%.6g)
    logv: %.6g(≈%.6g)
    ll: %.6g
`,
			when,
			x[0], sampleMean,
			x[1], sampleLogv,
			ll)
	}

	printState("Initially")

	// Run the optimizer
	opt := &infer.Grad{
		Rate:  RATE,
		Decay: DECAY,
	}
	for iter := 0; iter != NITER; iter++ {
		opt.Step(m, x)
	}

	ll = m.Observe(x)
	printState("Finally")
}

package main

import (
	"bitbucket.org/dtolpin/infergo/infer"
    . "bitbucket.org/dtolpin/infergo/examples/gmm/model/ad"
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
	NCOMP int = 2
	RATE  float64 = 0.01
	DECAY float64 = 0.995
	NITER int     = 100
)

func init() {
	flag.Usage = func() {
		log.Printf(`Gaussian mixture model:
		gmm [OPTIONS]`+"\n")
		flag.PrintDefaults()
	}
	flag.IntVar(&NCOMP, "ncomp", NCOMP, "number of components")
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
			log.Fatalf("Cannot open data file: %v", fname, err)
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
			1.899, -1.11, -0.9068, 1.291, -0.755,
			-0.4422, -0.144, 1.214, -0.8183, -0.3386,
			0.3863, -1.036, -0.6248, 1.014, 1.336,
			-1.487, 0.8223, -0.4268, 0.6754, 0.6206,
		}
	}

	// Define the problem
	m := &Model{Data: data, NComp: NCOMP}
	x := make([]float64, 2*m.NComp)

	// set a starting  point
	if m.NComp == 1 {
		x[0] = 0.
		x[1] = 1.
	} else {
		// Spread the initial components wide and thin
		for j := 0; j != m.NComp; j++ {
			x[2*j] = -2. + 4./float64(m.NComp - 1) * float64(j)
			x[2*j + 1] = 1.
		}
	}

	// Run the optimizer
	opt := &infer.GD {
		Rate: -RATE,
		Decay: DECAY,
	}
	for iter := 0; iter != NITER; iter++ {
		opt.Step(m, x)
	}

	// Print the result.
	log.Printf("Components:\n")
	for j := 0; j != m.NComp; j++ {
		log.Printf("\t%d: 𝜇=%.4g, 𝜎²=%.4g\n",
			j, x[2*j], math.Exp(x[2*j + 1]))
	}
}

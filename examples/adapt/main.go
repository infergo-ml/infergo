package main

import (
	. "bitbucket.org/dtolpin/infergo/examples/gmm/model/ad"
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
	NCOMP = 2
	STEP  = 0.1
	NITER = 100
	NBURN = 0
	NADPT = 10
	DEPTH = 3.
	RATE = 0.01
)

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
	flag.Usage = func() {
		log.Printf(`Gaussian mixture model:
		gmm [OPTIONS]` + "\n")
		flag.PrintDefaults()
	}
	flag.IntVar(&NCOMP, "ncomp", NCOMP, "number of components")
	flag.Float64Var(&STEP, "step", STEP, "NUTS step")
	flag.IntVar(&NITER, "niter", NITER, "number of iterations")
	flag.IntVar(&NBURN, "nburn", NBURN, "number of burned iterations")
	flag.IntVar(&NADPT, "nadpt", NADPT, "number of steps per adaptation")
	flag.Float64Var(&DEPTH, "depth", DEPTH, "target NUTS tree depth")
	flag.Float64Var(&RATE, "rate", RATE, "adaptation rate")
	log.SetFlags(0)
}

func main() {
	flag.Parse()
	if NBURN == 0 {
		NBURN = NITER
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
			1.899, -1.11, -0.9068, 1.291, -0.755,
			-0.4422, -0.144, 1.214, -0.8183, -0.3386,
			0.3863, -1.036, -0.6248, 1.014, 1.336,
			-1.487, 0.8223, -0.4268, 0.6754, 0.6206,
		}
	}

	// Define the problem
	m := &Model{Data: data, NComp: NCOMP}
	x := make([]float64, 2*m.NComp)

	// Set a starting  point
	if m.NComp == 1 {
		x[0] = 0.
		x[1] = 1.
	} else {
		// Spread the initial components wide and thin
		for j := 0; j != m.NComp; j++ {
			x[2*j] = -2. + 4./float64(m.NComp-1)*float64(j)
			x[2*j+1] = 1.
		}
		for k := 0; k != len(x); k++ {
			x[k] += 0.1 * rand.NormFloat64()
		}
	}

	// Let's infer the posterior with NUTS.
	nuts := &infer.NUTS{
		Eps: STEP / math.Sqrt(float64(len(m.Data))),
	}
	samples := make(chan []float64)
	nuts.Sample(m, x, samples)

	// Adapt toward optimum tree depth.
	da := &infer.DepthAdapter{
		DualAveraging: infer.DualAveraging{Rate: RATE},
		// Depth: DEPTH,
		// NAdpt: NADPT,
	}
	da.Adapt(nuts, samples, NBURN)

	// Collect after burn-in
	mean := make([]float64, m.NComp)
	stddev := make([]float64, m.NComp)
	n := 0.
	for i := 0; i != NITER; i++ {
		x := <-samples
		if len(x) == 0 {
			break
		}
		for j := 0; j != m.NComp; j++ {
			mean[j] += x[2*j]
			stddev[j] += math.Exp(0.5 * x[2*j+1])
		}
		n++
	}
	for j := 0; j != m.NComp; j++ {
		mean[j] /= n
		stddev[j] /= n
	}
	nuts.Stop()
	log.Printf("Components:\n")
	for j := 0; j != m.NComp; j++ {
		log.Printf("\t%d: mean=%.4g, stddev=%.4g\n",
			j, mean[j], stddev[j])
	}

	log.Printf(`NUTS:
	accepted: %d
	rejected: %d
	rate: %.4g
	mean depth: %.4g
`,
		nuts.NAcc, nuts.NRej,
		float64(nuts.NAcc)/float64(nuts.NAcc+nuts.NRej),
		nuts.MeanDepth())
}

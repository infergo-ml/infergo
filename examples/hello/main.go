package main

import (
	"bitbucket.org/dtolpin/infergo/ad"
	. "bitbucket.org/dtolpin/infergo/examples/hello/model/ad"
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
	STEP  float64 = 0.01
	DECAY float64 = 0.995
	NITER int     = 100
)

func init() {
	flag.Usage = func() {
		log.Printf(`Inferring parameters of the normal distribution:
		hello [OPTIONS] [data.csv]%s`, "\n")
		flag.PrintDefaults()
	}
	flag.Float64Var(&MEAN, "mean", MEAN, "initial mean")
	flag.Float64Var(&LOGV, "logv", LOGV, "initial log var")
	flag.Float64Var(&STEP, "step", STEP, "step size")
	flag.Float64Var(&DECAY, "decay", DECAY, "step decay")
	flag.IntVar(&NITER, "niter", NITER, "number of iterations")
	log.SetFlags(0)
}

func main() {
	flag.Parse()

	if flag.NArg() > 1 {
		log.Fatalf("unexpected positional arguments: %v",
			flag.Args()[1:])
	}

	var data []float64
	if flag.NArg() == 1 {
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
		data = []float64{
			-0.854, 1.067, -1.220, 0.818, -0.749,
			0.805, 1.443, 1.069, 1.426, 0.308}
	}
	m := &Model{Data: data}

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

	ll := m.Observe([]float64{MEAN, LOGV})
	grad := ad.Gradient()

	mean := MEAN
	logv := LOGV

	printState := func(when string) {
		log.Printf(`
%s:
    mean: %.6g(≈%.6g)
    logv: %.6g(≈%.6g)
    ll: %.6g
    grad: %.6g, %.6g
`,
			when,
			mean, sampleMean,
			logv, sampleLogv,
			ll,
			grad[0], grad[1])
	}

	printState("Initially")
	step := STEP
	for iter := 0; iter != NITER; iter++ {
		mean += step * grad[0]
		logv += step * grad[1]
		step *= DECAY
		ll = m.Observe([]float64{mean, logv})
		grad = ad.Gradient()
	}
	printState("Finally")

}

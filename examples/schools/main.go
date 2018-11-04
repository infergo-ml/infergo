package main

import (
	"bitbucket.org/dtolpin/infergo/ad"
	. "bitbucket.org/dtolpin/infergo/examples/schools/model/ad"
	"bitbucket.org/dtolpin/infergo/infer"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"
)

// Command line arguments

var (
	RATE  float64 = 0.01
	DECAY float64 = 0.9998
	MOMENTUM float64 = 0.5
	NITER int     = 10000
	LOGVTAU float64 = 1.
	LOGVETA float64 = 1.
)

func init() {
	flag.Usage = func() {
		fmt.Printf(`Eight school example. Usage:
		schools [OPTIONS]` + "\n")
		flag.PrintDefaults()
	}
	flag.Float64Var(&RATE, "rate", RATE, "learning rate")
	flag.Float64Var(&DECAY, "decay", DECAY, "rate decay")
	flag.Float64Var(&MOMENTUM, "momentum", MOMENTUM,
		"gradient momentum")
	flag.IntVar(&NITER, "niter", NITER, "number of iterations")
	flag.Float64Var(&LOGVTAU, "logvtau", LOGVTAU,
		"log variance of tau prior")
	flag.Float64Var(&LOGVETA, "logveta", LOGVETA,
		"log variance of eta priors")
	rand.Seed(time.Now().UTC().UnixNano())
}

func main() {
	flag.Parse()

	if flag.NArg() > 0 {
		fmt.Printf("unexpected positional arguments: %v",
			flag.Args())
		os.Exit(1)
	}

	// Define the problem
	m := &Model{
		J:     8,
		Y:     []float64{28, 8, -3, 7, -1, 1, 18, 12},
		Sigma: []float64{15, 10, 16, 11, 9, 11, 10, 18},
		LogVtau: LOGVTAU,
		LogVeta: LOGVETA,
	}
	x := make([]float64, 2+m.J)

	// Set a starting point
	x[0] = rand.NormFloat64()
	x[1] = rand.NormFloat64()
	for i := 2; i != len(x); i++ {
		x[i] = rand.NormFloat64()
	}
	// Compute log-likelihood of the starting point,
	// for comparison.
	ll0 := m.Observe(x)
	ad.Pop()

	// Run the optimizer
	opt := &infer.Grad{
		Rate:     RATE,
		Decay:    DECAY,
		Momentum: MOMENTUM,
	}
	for iter := 0; iter != NITER; iter++ {
		opt.Step(m, x)
	}

	mu := x[0]
	tau := math.Exp(x[1])
	eta := x[2:]
	fmt.Printf("Finally:\n\tmu=%.4g\n\ttau=%.4g\n\teta=", mu, tau)
	for _, eta := range eta {
		fmt.Printf("%.4g ", eta)
	}
	fmt.Printf("\n\ttheta(Y)=")
	for i, eta := range eta {
		fmt.Printf("%.4g(%.4g±%.4g) ", mu + tau*eta, m.Y[i], m.Sigma[i])
	}
	fmt.Printf("\n")
	ll := m.Observe(x)
	ad.Pop()
	fmt.Printf("Log-likelihood: %.4g ⇒ %.4g\n", ll0, ll)
}

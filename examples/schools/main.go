package main

import (
	. "bitbucket.org/dtolpin/infergo/examples/schools/model/ad"
	"bitbucket.org/dtolpin/infergo/infer"
	"bitbucket.org/dtolpin/infergo/ad"
	"flag"
	"fmt"
	"math"
	"os"
)

// Command line arguments

var (
	RATE  float64 = 0.01
	DECAY float64 = 0.997
	NITER int     = 1000
)

func init() {
	flag.Usage = func() {
		fmt.Printf(`Eight school example. Usage:
		schools [OPTIONS]` + "\n")
		flag.PrintDefaults()
	}
	flag.Float64Var(&RATE, "rate", RATE, "learning rate")
	flag.Float64Var(&DECAY, "decay", DECAY, "rate decay")
	flag.IntVar(&NITER, "niter", NITER, "number of iterations")
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
    }
    x := make([]float64, 2+m.J)

    // Set a starting point
    x[0] = 0.
    x[1] = 0.
    for i := 2; i != len(x); i++ {
        x[i] = 0.
    }
    // Compute fmt-likelihood of the starting point, 
    // for comparison.
    ll0 := m.Observe(x)
    ad.Pop()


    // Run the optimizer
    opt := &infer.Grad{
        Rate:  RATE,
        Decay: DECAY,
        Momentum: 0.5,
    }
    for iter := 0; iter != NITER; iter++ {
        opt.Step(m, x)
    }

    fmt.Printf("Finally:\n\tmu=%.4g\n\ttau=%.4g\n\teta=",
    x[0], math.Exp(x[1]))
    for _, eta := range x[2:] {
        fmt.Printf("%.4g ", eta)
    }
    fmt.Printf("\n")
    ll := m.Observe(x)
    ad.Pop()
    fmt.Printf("Log-likelihood: %.4g ⇒ %.4g\n", ll0, ll)
}

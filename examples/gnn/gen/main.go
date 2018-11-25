package main

import (
	"flag"
	"fmt"
	"math/rand"
	"time"
)

var (
	NCOMP = 3
	NOBS  = 100
	DIST  = 3.
)

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
	flag.IntVar(&NCOMP, "ncomp", NCOMP, "number of components")
	flag.IntVar(&NOBS, "nobs", NOBS, "number of observations")
	flag.Float64Var(&DIST, "dist", DIST, "distance between components")
}

func main() {
	flag.Parse()

	for i := 0; i != NOBS; i++ {
		icomp := rand.Intn(NCOMP)
		mu := DIST * (float64(icomp) - 0.5*float64(NCOMP-1))
		x := rand.NormFloat64() + mu
		fmt.Printf("%.4f,%d\n", x, icomp)
	}
}

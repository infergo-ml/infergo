package main

import (
    "flag"
    "log"
    "math"
    "bitbucket.org/dtolpin/infergo/ad"
    . "bitbucket.org/dtolpin/infergo/examples/normal/model/ad"
    
)

// Command line arguments

var (
    MEAN float64 = 0.
    LOGV float64 = 0.
)

func init() {
    flag.Float64Var(&MEAN, "mean", MEAN, "mean")
    flag.Float64Var(&LOGV, "logv", LOGV, "log variance")
    log.SetFlags(0)
}

func main () {
    flag.Parse()

    if flag.NArg() > 0 {
		log.Fatalf("unexpected position arguments: %v", flag.Args())
	}

    m := &Model {
        Data: []float64{
            -0.854, 1.067, -1.220, 0.818, -0.749,
             0.805, 1.443, 1.069, 1.426, 0.308},
    }

    s := 0.
    s2 := 0.
    for i := 0; i != len(m.Data); i++ {
        x := m.Data[i]
        s += x
        s2 += x*x
    }
    mean := s/float64(len(m.Data))
    logv := math.Log(s2/float64(len(m.Data)) - mean*mean)

    ll := m.Observe([]float64{MEAN, LOGV})
    grad := ad.Gradient()
    
    log.Printf("mean=%.6g(%.6g) logv=%.6g(%.6g) ll=%.6g grad=%.6g, %.6g\n", 
        MEAN, mean, LOGV, logv, ll, grad[0], grad[1])
}

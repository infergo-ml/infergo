# Infergo — Go programs that learn


[`infergo`](http://infergo.org/) is a  probabilistic
programming facility for the [Go language](http://golang.org/).
`infergo` allows to write probabilistic models in almost
unrestricted Go and relies on [automatic
differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)
for optimization and inference.

[![GoDoc](https://godoc.org/bitbucket.org/dtolpin/infergo?status.svg)](https://godoc.org/bitbucket.org/dtolpin/infergo)

MIT license (see [LICENSE](LICENSE))

## Example

\[[more examples](https://bitbucket.org/dtolpin/infergo/src/master/examples)\]

Learning parameters of the Normal distribution from
observations.

### Model

```Go
type Model struct {
    Data []float64
}

// x[0] is the mean, x[1] is the log stddev of the distribution
func (m *Model) Observe(x []float64) float64 {
    // Our prior is a unit normal ...
    ll := Normal.Logps(0, 1, x...)
    // ... but the posterior is based on data observations.
	ll += Normal.Logps(x[0], math.Exp(x[1]), m.Data...)
    return ll
}
```

### Inference

```Go
// Data
m := &Model{[]float64{
	-0.854, 1.067, -1.220, 0.818, -0.749,
	0.805, 1.443, 1.069, 1.426, 0.308}}

// Parameters
mean, logs := 0., 0.
x := []float64{mean, logs}
	
// Optimiziation
opt := &infer.Momentum{
    Rate:  0.01,
    Decay: 0.998,
}
for iter := 0; iter != 1000; iter++ {
    opt.Step(m, x)
}
mean, logs := x[0], x[1]

// Posterior
hmc := &infer.HMC{
	L:   10,
	Eps: 0.1,
}
samples := make(chan []float64)
hmc.Sample(m, x, samples)
for i := 0; i != 1000; i++ {
	x = <-samples
}
hmc.Stop()
```

## Acknowledgements

I owe a debt of gratitude to [Frank
Wood](https://www.cs.ubc.ca/~fwood/) who introduced me to
probabilistic programming and inspired me to pursue
probabilistic programming paradigms and applications. I also
want to thank [Jan-Willem van de
Meent](http://www.ccs.neu.edu/home/jwvdm/), with whom I had
fruitful discussions of motives, ideas, and implementation
choices behind `infergo`, and whose thoughts and recommendations
significantly influenced `infergo` design.  Finally, I want to
thank [PUB+](http://pubplus.com/), the company I work for, for
supporting me in development of `infergo` and letting me
experiment with applying probabilistic programming to critical
decision-making in production environment.

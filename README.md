# Learning programs in Go

Go as a platform for probabilistic inference. Uses
automatic differentiation and gradient descent
optimization.

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

# Learning programs in Go

Go as a platform for probabilistic inference. Uses
automatic differentiation and gradient descent
optimization.

MIT license (see [LICENSE](LICENSE))

## Example

\[[more examples](https://bitbucket.org/dtolpin/infergo/src/master/examples)\]

### Model

Learning parameters of the Normal distribution from
observations:

```Go
type Model struct {
    Data []float64
}

func (m *Model) Observe(x []float64) float64 {
    ll := 0.
    for i := 0; i != len(m.Data); i++ {
        ll += Normal.Logp(m.Data[i], x[0], x[1])
    }
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
mean, logv := 0., 0.
x := []float64{mean, logv}
	
// Optimiziation
opt := &infer.Momentum{
    Rate:  0.01,
    Decay: 0.998,
}
for iter := 0; iter != 1000; iter++ {
    opt.Step(m, x)
}
mean, logv := x[0], x[1]

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

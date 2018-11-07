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
package normal

import "math"

// The Data field is the observations
type Model struct {
    Data []float64
}

// x[0] is the mean, x[1] is the logvariance
func (m *Model) Observe(x []float64) float64 {
    mean, logv := x[0], x[1]
    vari := math.Exp(logv)
    ll := 0.
    for i := 0; i != len(m.Data); i++ {
        d := m.Data[i] - mean
        ll -= d*d/vari + logv
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
// mean ≈ 0.411, logv ≈ -0.117

// Parameters
mean, logv := 0., 0.
x := []float64{mean, logv}
	
// Optimiziation
opt := &infer.GD{
    Rate:  -RATE,
    Decay: DECAY,
}
for iter := 0; iter != NITER; iter++ {
    opt.Step(m, x)
}
mean, logv := x[0], x[1]

// Posterior
hmc := &infer.HMC{
	L:   NSTEPS,
	Eps: STEP,
}
samples := make(chan []float64)
hmc.Sample(m, x, samples)
for i := 0; i != NITER; i++ {
	x = <-samples
}
hmc.Stop()
```

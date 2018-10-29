# Learning programs in Go

Go as a platform for probabilistic inference. Uses
automatic differentiation and gradient descent
optimization.

MIT license (see [LICENSE](LICENSE))

## Example

### Model

```Go
// Learning the parameters of the Normal distribution from
// observations
package normal

import "math"

// The Data field is the observations
type Model struct {
    Data []float64
}

// x[0] is the mean, x[1] is the logvariance
func (m *Model) Observe(x []float64) float64 {
    mean := x[0]
    logv := x[1]
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
n := 100
step := 0.01
decay := 0.995

mean, logv := 0., 0.
for i := 0; i != n; i++ {
    m.Observe([]float64{mean, logv})
    grad := ad.Gradient()
    mean += step*grad[0]
    logv += step*grad[1]
    step *= decay
}
```

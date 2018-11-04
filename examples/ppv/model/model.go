// Determining the best bandwidth for page-per-visit prediction
// (http://dtolpin.github.io/posts/session-depth/)
package model

import (
	"math"
)

// data are the observations
type Model struct {
	PPV  []int
	NPages int
	PriorBandwidth float64
}

func (m *Model) Observe(x []float64) float64 {
	bandwidth := math.Exp(x[0])
	beliefs := make([][2]float64, m.NPages)
	churn_probability := 2. / float64(m.NPages)

	// initialize the beliefs
	for j := 0; j != m.NPages; j++ {
		beliefs[j][0] = 2. * churn_probability
		beliefs[j][1] = 2. * (1 - churn_probability)
	}

    // put a prior on the bandwidth
	target :=  -bandwidth / m.PriorBandwidth

	for _, ppv := range m.PPV {
		for j := 0; j != m.NPages; j++ {
			churned := j == ppv - 1

            // observe the ppv and update the belief
			evidence := beliefs[j][0] + beliefs[j][1]
			if churned {
				target += math.Log(beliefs[j][0] / evidence)
				beliefs[j][0] += 1.
			} else {
				target += math.Log(beliefs[j][1] / evidence)
				beliefs[j][1] += 1.
			}

			// discount the beliefs based on the bandwidth
			if evidence >= bandwidth  {
				discount := bandwidth / evidence
				beliefs[j][0] *= discount
				beliefs[j][1] *= discount
			}

            if churned {
                break
			}
        }
    }
	return target
}

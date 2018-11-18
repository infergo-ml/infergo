package model

import (
	"bitbucket.org/dtolpin/infergo/ad"
	. "bitbucket.org/dtolpin/infergo/dist/ad"
	"math"
)

type Model struct {
	Data []float64
}

func (m *Model) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	return ad.Return(ad.Call(func(_ []float64) {
		Normal.Logps(0, 0, m.Data...)
	}, 2, &x[0], ad.Elemental(math.Exp, &x[1])))
}

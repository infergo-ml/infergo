package model

import (
	. "bitbucket.org/dtolpin/infergo/dist/ad"
	"bitbucket.org/dtolpin/infergo/ad"
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
	return ad.Return(ad.Call(func(_vararg []float64) {
		Normal.Logp(0, 0, m.Data...)
	}, 2, &x[0], &x[1]))
}

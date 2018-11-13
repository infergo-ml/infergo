package model

import (
	. "bitbucket.org/dtolpin/infergo/dist/ad"
	"bitbucket.org/dtolpin/infergo/ad"
	"math"
)

type Model struct {
	PPV		[]int
	NPages		int
	PriorBandwidth	float64
}

func (m *Model) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	var bandwidth float64
	ad.Assignment(&bandwidth, ad.Elemental(math.Exp, &x[0]))
	var beliefs [][2]float64

	beliefs = make([][2]float64, m.NPages)
	var churn_probability float64
	ad.Assignment(&churn_probability, ad.Arithmetic(ad.OpDiv, ad.Value(2.), ad.Value(float64(m.NPages))))

	for j := 0; j != m.NPages; j = j + 1 {
		ad.Assignment(&beliefs[j][0], ad.Arithmetic(ad.OpMul, ad.Value(2.), &churn_probability))
		ad.Assignment(&beliefs[j][1], ad.Arithmetic(ad.OpMul, ad.Value(2.), (ad.Arithmetic(ad.OpSub, ad.Value(1), &churn_probability))))
	}
	var target float64
	ad.Assignment(&target, ad.Call(func(_vararg []float64) {
		Expon.Logp(0, 0)
	}, 2, ad.Arithmetic(ad.OpDiv, ad.Value(1), &m.PriorBandwidth), &bandwidth))

	for _, ppv := range m.PPV {
		for j := 0; j != m.NPages; j = j + 1 {
			var churned bool

			churned = j == ppv-1
			var evidence float64
			ad.Assignment(&evidence, ad.Arithmetic(ad.OpAdd, &beliefs[j][0], &beliefs[j][1]))
			if churned {
				ad.Assignment(&target, ad.Arithmetic(ad.OpAdd, &target, ad.Elemental(math.Log, ad.Arithmetic(ad.OpDiv, &beliefs[j][0], &evidence))))
				ad.Assignment(&beliefs[j][0], ad.Arithmetic(ad.OpAdd, &beliefs[j][0], ad.Value(1.)))
			} else {
				ad.Assignment(&target, ad.Arithmetic(ad.OpAdd, &target, ad.Elemental(math.Log, ad.Arithmetic(ad.OpDiv, &beliefs[j][1], &evidence))))
				ad.Assignment(&beliefs[j][1], ad.Arithmetic(ad.OpAdd, &beliefs[j][1], ad.Value(1.)))
			}

			if evidence >= bandwidth {
				var discount float64
				ad.Assignment(&discount, ad.Arithmetic(ad.OpDiv, &bandwidth, &evidence))
				ad.Assignment(&beliefs[j][0], ad.Arithmetic(ad.OpMul, &beliefs[j][0], &discount))
				ad.Assignment(&beliefs[j][1], ad.Arithmetic(ad.OpMul, &beliefs[j][1], &discount))
			}

			if churned {
				break
			}
		}
	}
	return ad.Return(&target)
}

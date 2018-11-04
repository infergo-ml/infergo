// The eight schools example as appears in PyStan documentation (and
// taken from "Bayesian Data Analysis", Section 5.5 by Gelman et al.i):
//
// data {
//     int<lower=0> J; // number of schools
//     vector[J] y; // estimated treatment effects
//     vector<lower=0>[J] sigma; // s.e. of effect estimates
// }
// parameters {
//     real mu;
//     real<lower=0> tau;
//     vector[J] eta;
// }
// transformed parameters {
//     vector[J] theta;
//     theta = mu + tau * eta;
// }
// model {
//     eta ~ normal(0, 1);
//     y ~ normal(theta, sigma);
// }
//
package model

import (
	"bitbucket.org/dtolpin/infergo/dist/ad"
	"math"
)

type Model struct {
	J     int       // number of schools
	Y     []float64 // estimated treatment effects
	Sigma []float64 // s.e. of effect estimates
	LogVtau, LogVeta float64 // log variances of tau and eta priors
}

func (m *Model) Observe(x []float64) float64 {
	ll := 0.0

	//  There are m.J + 2 parameters:
	// mu, logtau, eta[J]
	mu := x[0]
	ll += dist.Normal{0.}.Pdf(x[1], m.LogVtau)
	tau := math.Exp(x[1])
	eta := x[2:]

	for i, y := range m.Y {
		ll += dist.Normal{0}.Pdf(eta[i], m.LogVeta) 
		theta := mu + tau*eta[i]
		logVtheta := math.Log(m.Sigma[i] * m.Sigma[i])
		ll += dist.Normal{y}.Pdf(theta, logVtheta)
	}
	return ll
}

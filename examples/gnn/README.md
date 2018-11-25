# Gaussian mixture model

Inferring gaussian mixture components; both component parameters
and probabilities of data points to belong to each component are
inferred. First, a MLE estimate of the parameters is computed,
then, a Hamiltonian Monte Carlo variant is used to infer the
posterior, and empirical means of the full posterior are shown.

The default prior on component membership is improper uniform,
and HMC or NUTS are unlikely to converge with these settings.
Change to '-alpha 0.1 -tau 1' to see the inference converging.

## Installation

* [Install Go](https://golang.org/doc/install), version 1.11 or
  newer is required because the case study uses modules.

* Clone the repository.

```bash
git clone https://git@bitbucket.org/dtolpin/infergo-studies
```

* Change the current directory to `infergo-studies/gmm`.

## Running the inference

* Run `make`. Make will build the executable and run the
  inference on a small embedded data set, for self-check.
* Run `./gmm -niter 1000 -ncomp data-3.csv` for a bigger data
  set.

## Helpers

Subfolder `gen` contains a program for generating datasets. Run

```bash
go build -o generate ./gen
```

to build the program, and

```bash
./generate -ncomp 3 -nobs 1000 > my-dataset.csv
```

to generate a dataset with 3 components and 1000 observations.

## Pre-generate data

For data sets, for 2 and 3 components:

* data-2-broad.csv - -ncomp 2 -dist 3
* data-2-dense.csv - -ncomp 2 -dist 1
* data-3-broad.csv - -ncomp 3 -dist 3
* data-3-dense.csv - -ncomp 3 -dist 1

Dense data sets are more challenging for inference, and have greater
discrepancy between the mode and the mean.
	
## Example run

```text
$ ./gmm -alpha 0.1 -tau 1.
MLE (after 18 iterations):
* Log-likelihood: -112.259 => -88.787
* Components:
	0: mean=-0.834, stddev=0.402
	1: mean=1.182, stddev=0.458
* Observations:
    #	  value	 label	   p0	   p1
    0	  1.899	   1	  0.187	  0.813
    1	 -1.110	   0	  0.841	  0.159
    2	 -0.907	   0	  0.847	  0.153
    3	  1.291	   1	  0.163	  0.837
    4	 -0.755	   0	  0.851	  0.149
    5	 -0.442	   0	  0.855	  0.145
    6	 -0.144	   0	  0.809	  0.191
    7	  1.214	   1	  0.159	  0.841
    8	 -0.818	   0	  0.849	  0.151
    9	 -0.339	   0	  0.851	  0.149
   10	  0.386	   1	  0.149	  0.851
   11	 -1.036	   0	  0.843	  0.157
   12	 -0.625	   0	  0.854	  0.146
   13	  1.014	   1	  0.152	  0.848
   14	  1.336	   1	  0.164	  0.836
   15	 -1.487	   0	  0.827	  0.173
   16	  0.822	   1	  0.146	  0.854
   17	 -0.427	   0	  0.855	  0.145
   18	  0.675	   1	  0.143	  0.857
   19	  0.621	   1	  0.143	  0.857

Posterior means:
* HMC:
		accepted: 1427
		rejected: 574
		rate: 0.7131
	* Components:
	0: mean=-0.731, stddev=0.392
	1: mean=1.021, stddev=0.476
* Observations:
    #	  value	 label	    p0	    p1
    0	  1.899	   1	  0.167	  0.833
    1	 -1.110	   0	  0.941	  0.059
    2	 -0.907	   0	  0.759	  0.241
    3	  1.291	   1	  0.136	  0.864
    4	 -0.755	   0	  0.795	  0.205
    5	 -0.442	   0	  0.842	  0.158
    6	 -0.144	   0	  0.871	  0.129
    7	  1.214	   1	  0.419	  0.581
    8	 -0.818	   0	  0.891	  0.109
    9	 -0.339	   0	  0.821	  0.179
   10	  0.386	   1	  0.243	  0.757
   11	 -1.036	   0	  0.426	  0.574
   12	 -0.625	   0	  0.616	  0.384
   13	  1.014	   1	  0.224	  0.776
   14	  1.336	   1	  0.331	  0.669
   15	 -1.487	   0	  0.737	  0.263
   16	  0.822	   1	  0.192	  0.808
   17	 -0.427	   0	  0.602	  0.398
   18	  0.675	   1	  0.276	  0.724
   19	  0.621	   1	  0.182	  0.818
```

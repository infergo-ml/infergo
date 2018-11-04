all: build

PACKAGES=ad model infer mathx cmd/deriv

test:
	for package in $(PACKAGES); do go test ./$$package; done

build: test
	for package in $(PACKAGES); do go build ./$$package; done

GOFILES=ad/ad.go ad/elementals.go ad/tape.go \
	model/model.go \
	infer/infer.go

install: all test
	for package in $(PACKAGES); do go install ./$$package; done

# Examples
#
# Probabilistic Hello Wolrd: Inferring parameters of normal 
# distribution
.PHONY: hello
hello: 
	(cd examples/hello && make)

# Gaussian mixture model
.PHONY: gmm
gmm: 
	(cd examples/gmm && make)

#  8 schools
.PHONY: schools
schools: 
	(cd examples/schools && make)

#  pages per visit
.PHONY: ppv
ppv: 
	(cd examples/ppv && make)

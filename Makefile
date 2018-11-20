all: build

TESTPACKAGES=ad model infer mathx dist cmd/deriv
PACKAGES=$(TESTPACKAGES) dist/ad

EXAMPLES=hello gmm adapt schools ppv

examples: build $(EXAMPLES)

dist/ad/dist.go: dist/dist.go
	./deriv dist

test: 
	for package in $(TESTPACKAGES); do go test -short ./$$package; done

fulltest:
	for package in $(TESTPACKAGES); do go test -count=1 ./$$package; done

build: test
	for package in $(PACKAGES); do go build ./$$package; done

benchmark: test
	for package in $(TESTPACKAGES); do go test -bench . ./$$package; done

GOFILES=ad/ad.go ad/elementals.go ad/tape.go \
	model/model.go \
	infer/infer.go

install: all test dist/ad/dist.go
	for package in $(PACKAGES); do go install ./$$package; done
	if [ -n "$(GOPATH)" ] ; then cp deriv $(GOPATH)/bin ; fi

clean-examples:
	for x in $(EXAMPLES); do (cd examples/$$x && make clean); done

clean: clean-examples
	rm -rf deriv dist/ad

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

# NUTS Step adaptation 
.PHONY: adapt
adapt:
	(cd examples/adapt && make)

#  8 schools
.PHONY: schools
schools:
	(cd examples/schools && make)

#  pages per visit
.PHONY: ppv
ppv:
	(cd examples/ppv && make)

all: build

PACKAGES=ad model infer mathx dist dist/ad cmd/deriv
EXAMPLES=hello gmm adapt schools ppv

examples: build $(EXAMPLES)

dist/ad/dist.go: dist/dist.go
	deriv dist

test: dist/ad/dist.go
	for package in $(PACKAGES); do go test -short ./$$package; done

fulltest: test
	for package in $(PACKAGES); do go test ./$$package; done

build: test
	for package in $(PACKAGES); do go build ./$$package; done

benchmark: test
	for package in $(PACKAGES); do go test -bench . ./$$package; done

GOFILES=ad/ad.go ad/elementals.go ad/tape.go \
	model/model.go \
	infer/infer.go

install: all test
	for package in $(PACKAGES); do go install ./$$package; done


clean-examples:
	for x in $(EXAMPLES); do (cd examples/$$x && make clean); done

clean: clean-examples
	rm -f deriv

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

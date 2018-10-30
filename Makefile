all: test deriv

.PHONY: deriv

PACKAGES=ad model infer mathx

test:
	for package in $(PACKAGES); do go test ./$$package; done

GOFILES=ad/ad.go ad/elementals.go ad/tape.go \
	model/model.go \
	infer/infer.go

deriv: $(GOFILES)
	go build deriv.go

install: all test
	for package in $(PACKAGES); do go install ./$$package; done
	go install deriv.go

clean:
	rm -f ./deriv

# Examples
#
# Probabilistic Hello Wolrd: Inferring parameters of normal 
# distribution
.PHONY: hello
hello: 
	(cd examples/hello && make)

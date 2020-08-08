all: build

GO=go

TESTPACKAGES=ad model infer mathx dist cmd/deriv
PACKAGES=$(TESTPACKAGES) dist/ad

EXAMPLES=hello gmm adapt schools ppv

examples: build $(EXAMPLES)

test: dist/ad/dist.go
	for package in $(TESTPACKAGES); do go test ./$$package; done

dist/ad/dist.go: dist/dist.go
	$(GO) build ./cmd/deriv
	./deriv dist

build: test
	for package in $(PACKAGES); do $(GO) build ./$$package; done

benchmark: test
	for package in $(TESTPACKAGES); do $(GO) test -bench . ./$$package; done

GOFILES=ad/ad.go ad/elementals.go ad/tape.go \
	model/model.go \
	infer/infer.go

install: all test
	for package in $(PACKAGES); do $(GO) install ./$$package; done

push:
	for repo in origin ssh://git@github.com/infergo-ml/infergo; do git push $$repo; git push --tags $$repo; done

clean-examples:
	for x in $(EXAMPLES); do (cd examples/$$x && make clean); done

clean: clean-examples
	rm -rf deriv

# Examples
#
# Probabilistic Hello Wolrd: Inferring parameters of normal
# distribution
.PHONY: hello
hello:
	(cd examples/hello && make GO=$(GO))

# Gaussian mixture model
.PHONY: gmm
gmm:
	(cd examples/gmm && make GO=$(GO))

# NUTS Step adaptation 
.PHONY: adapt
adapt:
	(cd examples/adapt && make GO=$(GO))

#  8 schools
.PHONY: schools
schools:
	(cd examples/schools && make GO=$(G0))

#  pages per visit
.PHONY: ppv
ppv:
	(cd examples/ppv && make GO=$(GO))

#  multi-threaded hello world
.PHONY: mt
mt:
	(cd examples/mt && make GO=$(GO))

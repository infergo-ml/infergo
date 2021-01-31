all: adapt

GO=go
DERIV="../../deriv"

adapt: main.go
	(cd ../gmm && make)
	$(GO) build .
	./adapt

clean:
	rm -f ./adapt

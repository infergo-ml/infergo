all: adapt

DERIV="../../deriv"

adapt: main.go
	(cd ../gmm && make)
	go build .
	./adapt

clean:
	rm -f ./adapt

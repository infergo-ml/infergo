all: hello

GO=go
DERIV="../../deriv"

hello: model/ad/model.go main.go
	$(GO) build .
	./hello

model/ad/model.go: model/model.go
	$(DERIV) model

clean:
	rm -f ./hello model/ad/*.go

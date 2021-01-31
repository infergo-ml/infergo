all: mt

GO=go
DERIV="../../deriv"

mt: model/ad/model.go main.go
	$(GO) build .
	./mt

model/ad/model.go: model/model.go
	$(DERIV) model

clean:
	rm -f ./mt model/ad/*.go

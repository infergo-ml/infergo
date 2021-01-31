all: schools

GO=go
DERIV="../../deriv"

schools: model/ad/model.go main.go
	$(GO) build .
	./schools

model/ad/model.go: model/model.go
	$(DERIV) model

clean:
	rm -f ./schools model/ad/*.go

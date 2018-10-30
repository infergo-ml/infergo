all: gmm

DERIV="../../deriv"

gmm: model/ad/model.go main.go
	go build .
	./gmm

model/ad/model.go: model/model.go
	$(DERIV) model

clean:
	rm -f ./gmm model/ad/*.go

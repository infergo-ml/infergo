all: schools

DERIV="../../deriv"

schools: model/ad/model.go main.go
	go build .
	./schools

model/ad/model.go: model/model.go
	$(DERIV) model

clean:
	rm -f ./schools model/ad/*.go

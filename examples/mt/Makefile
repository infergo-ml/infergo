all: mt

DERIV="../../deriv"

mt: model/ad/model.go main.go
	go build .
	./mt

model/ad/model.go: model/model.go
	$(DERIV) model

clean:
	rm -f ./mt model/ad/*.go

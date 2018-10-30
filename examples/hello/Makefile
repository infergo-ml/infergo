all: hello

DERIV="../../deriv"

hello: model/ad/model.go main.go
	go build .
	./hello

model/ad/model.go: model/model.go
	$(DERIV) model

clean:
	rm -f ./hello

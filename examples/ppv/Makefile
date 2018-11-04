all: goppv

goppv: model/ad/model.go main.go
	go build .
	./goppv

model/ad/model.go: model/model.go
	deriv model

clean:
	rm -f ./goppv model/ad/*.go

all: ppv

ppv: model/ad/model.go main.go
	go build .
	./ppv

model/ad/model.go: model/model.go
	deriv model

clean:
	rm -f ./ppv model/ad/*.go

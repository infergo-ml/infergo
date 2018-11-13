all: ppv

DERIV="../../deriv"

ppv: model/ad/model.go main.go
	go build .
	./ppv

model/ad/model.go: model/model.go
	$(DERIV) model

clean:
	rm -f ./ppv model/ad/*.go

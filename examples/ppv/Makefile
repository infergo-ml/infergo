all: ppv

GO=go
DERIV="../../deriv"

ppv: model/ad/model.go main.go
	$(GO) build .
	./ppv

model/ad/model.go: model/model.go
	$(DERIV) model

clean:
	rm -f ./ppv model/ad/*.go

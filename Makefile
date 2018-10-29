all: deriv

.PHONY: deriv

PACKAGES=ad model infer

test:
	for package in $(PACKAGES); do go test ./$$package; done

deriv: test
	go build deriv.go

install: all
	for package in $(PACKAGES); do install ./$$package; done
	go install deriv.go

hello: deriv
	./deriv ./examples/hello/model
	go build ./examples/hello
	./hello

clean:
	-rm -rf deriv hello ./examples/hello/ad

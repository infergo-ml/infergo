all: deriv

.PHONY: deriv

deriv:
	go test ./...
	go build deriv.go

install: all
	go test ./...
	go install ./...
	go install deriv.go

clean:
	-rm -rf deriv

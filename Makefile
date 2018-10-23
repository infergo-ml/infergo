all: deriv

deriv: deriv.go
	go test ./...
	go build deriv.go

install: all
	go test ./...
	go install ./...
	go install deriv.go

clean:
	-rm -rf deriv

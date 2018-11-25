all: gnn generate

gnn: model/ad/model.go main.go
	go build .
	./gnn
	./gnn -alpha 0.1 -tau 1.

generate: gen/main.go
	go build -o $@ ./gen

model/ad/model.go: model/model.go
	deriv model

clean:
	rm -f ./gnn ./generate model/ad/*.go

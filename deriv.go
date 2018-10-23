package main

import (
	"bitbucket.org/dtolpin/infergo/ad"
	"flag"
)

var (
// command line flags
)

func main() {
	flag.Parse()
	if flag.NArg() == 0 {
		ad.Differentiate(".")
	} else {
		for i := 0; i != flag.NArg(); i++ {
			err := ad.Differentiate(flag.Arg(i))
			if err != nil {
				println(err.Error())
			}
		}
	}
}

package main

import (
	"bitbucket.org/dtolpin/infergo/ad"
	"flag"
	"log"
)

var (
// command line flags
)

func init() {
	log.SetFlags(0)
}

func main() {
	flag.Parse()
	if flag.NArg() == 0 {
		// If there are no command line arguments, differentiate
		// the current directory.
		deriv(".")
	} else {
		// Otherwise differentiate each directory given.
		for i := 0; i != flag.NArg(); i++ {
			deriv(flag.Arg(i))
		}
	}
}

// deriv differentiates the model in the package.
func deriv(model string) {
	err := ad.Deriv(model)
	if err != nil {
		log.Printf("ERROR: %v", err.Error())
	}
}
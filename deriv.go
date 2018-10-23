package main

import (
	"bitbucket.org/dtolpin/infergo/ad"
	"flag"
	"log"
	"os"
)

var (
// command line flags
)

// l is the global logger
var l *log.Logger

func init() {
	l = log.New(os.Stderr, "deriv: ", 0)
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
        l.Printf("ERROR: %v", err.Error())
    }
}

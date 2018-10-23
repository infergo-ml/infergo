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

func main() {
	l := log.New(os.Stderr, "deriv: ", 0)
	flag.Parse()
	if flag.NArg() == 0 {
		// If there are no command line arguments, differentiate
		// the current directory.
		ad.Differentiate(".")
	} else {
		// Otherwise differentiate each directory given.
		for i := 0; i != flag.NArg(); i++ {
			err := ad.Differentiate(flag.Arg(i))
			if err != nil {
				l.Printf("ERROR: %v", err.Error())
			}
		}
	}
}

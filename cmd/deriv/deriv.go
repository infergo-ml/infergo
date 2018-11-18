package main

import (
	"bitbucket.org/dtolpin/infergo/ad"
	"flag"
	"fmt"
	"log"
)

var (
	// command line flags
	PREFIX = "_"
)

func init() {
	flag.StringVar(&PREFIX, "prefix", PREFIX,
		"prefix of generated identifiers")
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(),
			`Generates a differentiated model. Usage:
    deriv [flags] [path/to/model/directory/]
If the path is omitted, the model in the current directory `+
				`is differentiated. The differentiated model is placed into `+
				`the 'ad/' subdirectory. Flags:
`)
		flag.PrintDefaults()
	}
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
	err := ad.Deriv(model, PREFIX)
	if err != nil {
		log.Printf("ERROR: %v", err.Error())
	}
}

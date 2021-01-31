package main

import (
	"bitbucket.org/dtolpin/infergo/ad"
	"flag"
	"fmt"
	"log"
)

const (
	command = "deriv"
	version = "0.9.2"
)

var (
	// command line flags
	VERSION = false
	PREFIX  = "_"
)

func init() {
	flag.BoolVar(&VERSION, "version", VERSION,
		"print version")
	flag.StringVar(&PREFIX, "prefix", PREFIX,
		"prefix of generated identifiers")
	flag.BoolVar(&ad.Fold, "fold", ad.Fold,
		"fold constants")
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(),
			`Generates a differentiated model. Usage:
    %s [flags] [path/to/model/directory/]
If the path is omitted, the model in the current directory `+
				`is differentiated. The differentiated model `+
				`is placed into the 'ad/' subdirectory. Flags:
`,
			command)
		flag.PrintDefaults()
	}
	log.SetFlags(0)
}

func main() {
	flag.Parse()
	if VERSION {
		fmt.Fprintf(flag.CommandLine.Output(), "infergo %s v%s\n", command, version)
		return
	}

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

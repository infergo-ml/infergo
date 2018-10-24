package ad

import (
	_ "bytes"
	"fmt"
	"go/token"
	"go/ast"
	"go/parser"
	"testing"
)

func Test(t *testing.T) {
}

// Tooling for comparing models

// The input to ad routines is a parsed package. Let's
// emulate parsing a package on scripts.
func parseTestModel(sources ...string) (
    fset *token.FileSet,
    pkg *ast.Package,
) {
    fset = token.NewFileSet()

    // parse it
    files := make(map[string]*ast.File)
    for i, source := range sources {
	fname := fmt.Sprintf("%v", i)
	file, err := parser.ParseFile(fset, fname, source, 0)
	if err != nil {
		panic(err)
	}
	files[fname] = file
    }
    pkg, err  := ast.NewPackage(
	fset,
	files,
	nil,
	ast.NewScope(nil))
    if  err != nil {
	panic(err)
    }

    return fset, pkg
}



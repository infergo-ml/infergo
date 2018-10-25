package ad

import (
	_ "bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

// Tooling for comparing models

// The input to ad routines is a parsed package. Let's
// emulate parsing a package on scripts.
func parseTestModel(sources []string) (
	fset *token.FileSet,
	pkg *ast.Package,
) {
	fset = token.NewFileSet()

	// parse it
	for i, source := range sources {
		fname := fmt.Sprintf("file_%v", i)
		if file, err := parser.ParseFile(
			fset, fname, source, 0); err == nil {
			name := file.Name.Name
			if pkg == nil {
				pkg = &ast.Package{
					Name:  name,
					Files: make(map[string]*ast.File),
				}
			}
			pkg.Files[fname] = file
		} else {
			panic(err)
		}
	}

	return fset, pkg
}

func TestCollectModelTypes(t *testing.T) {
	for _, c := range []struct {
		model []string
		types map[string]bool
	}{
		// Single model
		{[]string{
			`package single

type Model float64

func (m Model) Observe(x []float64) float64 {
	return - float64(m) * x[0]
}
`,
		},
			map[string]bool{
				"single.Model": true,
			}},

		// No model
		{[]string{
			`package nomodel

type Model float64

func (m Model) observe(x []float64) float64 {
	return - float64(m) * x[0]
}
`,
		},
			map[string]bool{}},

		// Two models
		{[]string{
			`package double

type ModelA float64
type ModelB float64

func (m ModelA) Observe(x []float64) float64 {
	return - float64(m) * x[0]
}

func (m ModelB) Observe(x []float64) float64 {
	return - float64(m) / x[0]
}
`,
		},
			map[string]bool{
				"double.ModelA": true,
				"double.ModelB": true,
			}},
	} {
		fset, pkg := parseTestModel(c.model)
		println(fset)
		println(pkg)
		println(fset)
		info, err := checkModel(pkg.Name, fset, pkg)
		if err != nil {
			t.Errorf("failed to check model %v: %s",
				pkg.Name, err)
		}
		modelTypes, err := collectModelTypes(fset, pkg, info)
		if len(modelTypes) > 0 && err != nil {
			// Ignore the error when there is no model
			panic(err)
		}
		for _, mt := range modelTypes {
			if !c.types[mt.String()] {
				t.Errorf("model %v: type %v is not a model",
					pkg.Name, mt)
			}
			delete(c.types, mt.String())
		}
		for k := range c.types {
			t.Errorf("model %v: type %v was not collected",
				pkg.Name, k)
		}
	}
}

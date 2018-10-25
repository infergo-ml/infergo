package ad

import (
	_ "bytes"
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

// Tooling for comparing models

// The input to ad routines is a parsed package. Let's
// emulate parsing a package on scripts.
func parseTestModel(m *model, sources map[string]string) {
	m.fset = token.NewFileSet()

	// parse it
	for fname, source := range sources {
		if file, err := parser.ParseFile(
			m.fset, fname, source, 0); err == nil {
			name := file.Name.Name
			if m.pkg == nil {
				m.pkg = &ast.Package{
					Name:  name,
					Files: make(map[string]*ast.File),
				}
			}
			m.pkg.Files[fname] = file
		} else {
			panic(err)
		}
	}
}

func TestCollectTypes(t *testing.T) {
	for _, c := range []struct {
		model map[string]string
		types map[string]bool
	}{
		// Single model
		{map[string]string{
            "single.go": `package single

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
		{map[string]string{
            "nomodel.go": `package nomodel

type Model float64

func (m Model) observe(x []float64) float64 {
	return - float64(m) * x[0]
}
`,
		},
			map[string]bool{}},

		// Two models
		{map[string]string{
            "double.go": `package double

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
		m := &model{}
		parseTestModel(m, c.model)
		err := m.check(m.pkg.Name)
		if err != nil {
			t.Errorf("failed to check model %v: %s",
				m.pkg.Name, err)
		}
		mtypes, err := m.collectTypes()
		if len(mtypes) > 0 && err != nil {
			// Ignore the error when there is no model
			panic(err)
		}
		for _, mt := range mtypes {
			if !c.types[mt.String()] {
				t.Errorf("model %v: type %v is not a model",
					m.pkg.Name, mt)
			}
			delete(c.types, mt.String())
		}
		for k := range c.types {
			t.Errorf("model %v: type %v was not collected",
				m.pkg.Name, k)
		}
	}
}

func TestCollectFiles(t *testing.T) {
	for _, c := range []struct {
		model map[string]string
		fnames map[string]bool
	}{
		// Single file, two methods
		{map[string]string{
            "single.go": `package single

type Model float64

func (m Model) Observe(x []float64) float64 {
	return - float64(m) * x[0]
}

func (m Model) Sample() float64 {
	return 0.0
}
`,
		},
			map[string]bool{
				"single.go": true,
			}},

		// Two files, first file contains the methods
		{map[string]string{
            "first.go": `package first

type Model float64

func (m Model) Observe(x []float64) float64 {
	return - float64(m) * x[0]
}
`,
            "second.go": `package first

func foo() {
}
`,
		},
			map[string]bool{
                "first.go": true,
            }},

		// Two files, first file contains the methods
		{map[string]string{
            "first.go": `package both

type Model float64

func (m *Model) Observe(x []float64) float64 {
	return - float64(*m) * x[0]
}
`,
            "second.go": `package both

func (m Model) Sample() float64 {
	return 0.0
}
`,
		},
			map[string]bool{
                "first.go": true,
                "second.go": true,
            }},
	} {
		m := &model{}
		parseTestModel(m, c.model)
		err := m.check(m.pkg.Name)
		if err != nil {
			t.Errorf("failed to check model %v: %s",
				m.pkg.Name, err)
		}
        mtypes, err := m.collectTypes()
		mfiles, err := m.collectFiles(mtypes)
		if len(mfiles) > 0 && err != nil {
			// Ignore the error when there is no model
			panic(err)
		}
		for _, mf := range mfiles {
            fname := m.fset.Position(mf.Package).Filename
			if !c.fnames[fname] {
				t.Errorf("model %v: file %q contains no methods",
					m.pkg.Name, fname)
			}
			delete(c.fnames, fname)
		}
		for k := range c.fnames {
			t.Errorf("model %v: file %q was not collected",
				m.pkg.Name, k)
		}
	}
}

func TestCollectMethods(t *testing.T) {
}

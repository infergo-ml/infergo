package ad

import (
	"bytes"
	"go/ast"
	"go/parser"
	"go/printer"
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
				m.path = name
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

// Tests that the expected source is equivalent to the tree.
func equiv(gotTree *ast.File, expected string) bool {
	// Normalize source code by parsing and printing.

	// Parse it
	fileSet := token.NewFileSet()
	expectedTree, error := parser.ParseFile(fileSet, "", expected, 0)
	if error != nil {
		panic(error)
	}

	// Print it
	gotBuffer := new(bytes.Buffer)
	expectedBuffer := new(bytes.Buffer)
	// We allocate a new file set to normalize whitespace
	fileSet = token.NewFileSet()
	printer.Fprint(gotBuffer, fileSet, gotTree)
	printer.Fprint(expectedBuffer, fileSet, expectedTree)

	// compare strings
	return gotBuffer.String() == expectedBuffer.String()
}

func TestEquiv(t *testing.T) {
	// kick the tyres
	for _, c := range []struct {
		got      string
		expected string
		equal    bool
	}{
		{
			"package a; func foo() {}",
			"package a\n\nfunc foo() {\n}",
			true,
		},
		{
			"package a\n\nvar x int = 1",
			"package a\n\nvar x int",
			false,
		},
	} {
		fileSet := token.NewFileSet()
		gotTree, error := parser.ParseFile(fileSet, "", c.got, 0)
		if error != nil {
			panic(error)
		}
		if equiv(gotTree, c.expected) != c.equal {
			t.Errorf("Sources\n---\n%v\n---\n"+
				"and\n---\n%v\n---\nshould %v.",
				c.got, c.expected,
				map[bool]string{
					false: "not be equal",
					true:  "be equal",
				}[c.equal])
		}
	}
}

// Tests of ad transformations, stage by stage

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
		err := m.check()
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

func TestCollectMethods(t *testing.T) {
	for _, c := range []struct {
		model  map[string]string
		mnames map[string]bool
	}{
		// Single file, single method
		{map[string]string{
			"one.go": `package single

type Model float64

func (m Model) Observe(x []float64) float64 {
	return - float64(m) * x[0]
}
`,
		},
			map[string]bool{
				"Observe": true,
			}},
		// Single file, two methods
		{map[string]string{
			"one.go": `package double

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
				"Observe": true,
				"Sample":  true,
			}},

		// Two files, two methods
		{map[string]string{
			"first.go": `package two

type Model float64

func (m Model) Observe(x []float64) float64 {
	return - float64(m) * x[0]
}
`,
			"second.go": `package two

func (m Model) Sample() float64 {
	return 0.0
}
`,
		},
			map[string]bool{
				"Observe": true,
				"Sample":  true,
			}},
	} {
		m := &model{}
		parseTestModel(m, c.model)
		err := m.check()
		if err != nil {
			t.Errorf("failed to check model %v: %s",
				m.pkg.Name, err)
		}
		mtypes, err := m.collectTypes()
		methods, err := m.collectMethods(mtypes)
		if len(methods) > 0 && err != nil {
			// Ignore the error when there is no model
			panic(err)
		}
		for _, method := range methods {
			mname := method.Name.Name
			if !c.mnames[mname] {
				t.Errorf("model %v: file %q contains no methods",
					m.pkg.Name, mname)
			}
			delete(c.mnames, mname)
		}
		for k := range c.mnames {
			t.Errorf("model %v: file %q was not collected",
				m.pkg.Name, k)
		}
	}
}

func TestCollectFiles(t *testing.T) {
	for _, c := range []struct {
		model  map[string]string
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
				"first.go":  true,
				"second.go": true,
			}},
	} {
		m := &model{}
		parseTestModel(m, c.model)
		err := m.check()
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

func TestSimplify(t *testing.T) {
	for _, c := range []struct {
		original, simplified string
	}{
		{`package define

type Model float64

func (m Model) Observe(x []float64) float64 {
    a := 0.
	b := []float64{1.}
    d, e := 3., 4.
	return a + b[0]  - d - e
}`,
			`package define

type Model float64

func (m Model) Observe(x []float64) float64 {
    var a float64
    a = 0.
    var b []float64
	b = []float64{1.}
    var d float64
    var e float64
    d, e = 3., 4.
	return a + b[0] - d - e
}`,
		},
		{`package opassign

type Model float64

func (m Model) Observe(x []float64) float64 {
    var a float64
    a += 2
    a -= 3.
    a *= 4.
    a /= 5.
	return a
}`,
			`package opassign

type Model float64

func (m Model) Observe(x []float64) float64 {
    var a float64
    a = a + 2
    a = a - 3.
    a = a * 4.
    a = a / 5.
	return a
}`,
		},
		{`package incdec

type Model float64

func (m Model) Observe(x []float64) float64 {
    a := 1.
    a++
    a--
	return a
}`,
			`package opassign

type Model float64

func (m Model) Observe(x []float64) float64 {
    var a float64
    a = 1.
    a = a + 1
    a = a - 1
	return a
}`,
		},
	} {
		m := &model{}
		parseTestModel(m, map[string]string{
			"original.go": c.original,
		})
		err := m.check()
		if err != nil {
			t.Errorf("failed to check model %v: %s",
				m.pkg.Name, err)
		}
		mtypes, err := m.collectTypes()
		methods, err := m.collectMethods(mtypes)
		for _, method := range methods {
			m.simplify(method)
		}
		if !equiv(m.pkg.Files["original.go"], c.simplified) {
			b := new(bytes.Buffer)
			printer.Fprint(b, m.fset, m.pkg.Files["original.go"])

			t.Errorf("model %v:\n---\n%v\n---\n"+
				" not equivalent to \n---\n%v\n---\n",
				m.pkg.Name,
				b.String(),
				c.simplified)
		}
	}
}

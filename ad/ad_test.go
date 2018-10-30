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
		err = m.collectTypes()
		if len(m.typs) > 0 && err != nil {
			// Ignore the error when there is no model
			panic(err)
		}
		for _, mt := range m.typs {
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
		err = m.collectTypes()
		methods, err := m.collectMethods()
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

func TestSimplify(t *testing.T) {
	for _, c := range []struct {
		original, simplified string
	}{
		//====================================================
		{`package vardecl

type Model float64

func (m Model) Observe(x []float64) float64 {
	var a, c, e = 0., 'c', 1
	var b float64
	println(c, e)
	return a + b
}`,
			//----------------------------------------------------
			`package vardecl

type Model float64

func (m Model) Observe(x []float64) float64 {
	var (
		a float64
		c rune
		e int
	)
	a, c, e = 0., 'c', 1
	var b float64
	println(c, e)
	return a + b
}`,
		},
		//====================================================
		{`package simplestmt

type Model float64

func foo() (int, interface{}) {return 0, &struct {}{}}

func (m Model) Observe(x []float64) float64 {
	if _, err := foo(); err != nil {
		println("error")
	}
	return 0.
}`,
			//----------------------------------------------------
			`package simplestmt

type Model float64

func foo() (int, interface{}) {return 0, &struct {}{}}

func (m Model) Observe(x []float64) float64 {
	if _, err := foo(); err != nil {
		println("error")
	}
	return 0.
}`,
		},
		//====================================================
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
			//----------------------------------------------------
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
		//====================================================
		{`package incdec

type Model float64

func (m Model) Observe(x []float64) float64 {
	a := 1.
	a++
	a--
	return a
}`,
			//----------------------------------------------------
			`package incdec

type Model float64

func (m Model) Observe(x []float64) float64 {
	var a float64
	a = 1.
	a = a + 1
	a = a - 1
	return a
}`,
		},
		//====================================================
		{`package plus

type Model float64

func (m Model) Observe(x []float64) float64 {
	a := 1.
	b := +a
	return b
}`,
			//----------------------------------------------------
			`package plus

type Model float64

func (m Model) Observe(x []float64) float64 {
	var a float64
	a = 1.
	var b float64
	b = a
	return b
}`,
			//====================================================
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
		err = m.collectTypes()
		methods, err := m.collectMethods()
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

func TestDifferentiate(t *testing.T) {
	for _, c := range []struct {
		original, differentiated string
	}{
		//====================================================
		{`package lit

type Model float64

func (m Model) Observe(x []float64) float64 {
	return 0
}

func (m Model) Count() int {
	return 0
}`,
			//----------------------------------------------------
			`package lit

import "bitbucket.org/dtolpin/infergo/ad"

type Model float64

func (m Model) Observe(x []float64) float64 {
	ad.Setup(x)
	return ad.Return(ad.Value(0))
}

func (m Model) Count() int {
	ad.Enter()
	return 0
}`,
		},
		//====================================================
		{`package ident

type Model float64

func (m Model) Observe(x []float64) float64 {
	y := 1.
	return y
}

func (m Model) Count() int {
	y := 1
	return y
}`,
			//----------------------------------------------------
			`package ident

import "bitbucket.org/dtolpin/infergo/ad"

type Model float64

func (m Model) Observe(x []float64) float64 {
	ad.Setup(x)
	var y float64
	ad.Assignment(&y, ad.Value(1.))
	return ad.Return(&y)
}

func (m Model) Count() int {
	ad.Enter()
	var y int
	y = 1
	return y
}`,
		},
		//====================================================
		{`package index

type Model float64

func (m Model) Observe(x []float64) float64 {
	return x[0]
}`,
			//----------------------------------------------------
			`package index

import "bitbucket.org/dtolpin/infergo/ad"

type Model float64

func (m Model) Observe(x []float64) float64 {
	ad.Setup(x)
	return ad.Return(&x[0])
}`,
		},
		//====================================================
		{`package selector

type Model struct {
	y float64
}

func (m Model) Observe(x []float64) float64 {
	return m.y
}`,
			//----------------------------------------------------
			`package selector

import "bitbucket.org/dtolpin/infergo/ad"

type Model struct {
	y float64
}

func (m Model) Observe(x []float64) float64 {
	ad.Setup(x)
	return ad.Return(&m.y)
}`,
		},
		//====================================================
		{`package star

type Model float64

func (m Model) Observe(x []float64) float64 {
	y := &x[0]
	return *y
}`,
			//----------------------------------------------------
			`package star

import "bitbucket.org/dtolpin/infergo/ad"

type Model float64

func (m Model) Observe(x []float64) float64 {
	ad.Setup(x)
	var y *float64
	y = &x[0]
	return ad.Return(y)
}`,
		},
		//====================================================
		{`package unary

type Model float64

func (m Model) Observe(x []float64) float64 {
	y := +x[0]
	y = -x[1]
	return y
}`,
			//----------------------------------------------------
			`package unary

import "bitbucket.org/dtolpin/infergo/ad"

type Model float64

func (m Model) Observe(x []float64) float64 {
	ad.Setup(x)
	var y float64
	ad.Assignment(&y, &x[0])
	ad.Assignment(&y, ad.Arithmetic(ad.OpNeg, &x[1]))
	return ad.Return(&y)
}`,
		},
		//====================================================
		{`package binary

type Model float64

func (m Model) Observe(x []float64) float64 {
	y := x[0] + x[1]
	y = y - x[2]
	y = y * x[3]
	return y / x[4]
}`,
			//----------------------------------------------------
			`package binary

import "bitbucket.org/dtolpin/infergo/ad"

type Model float64

func (m Model) Observe(x []float64) float64 {
	ad.Setup(x)
	var y float64
	ad.Assignment(&y, ad.Arithmetic(ad.OpAdd, &x[0], &x[1]))
	ad.Assignment(&y, ad.Arithmetic(ad.OpSub, &y, &x[2]))
	ad.Assignment(&y, ad.Arithmetic(ad.OpMul, &y, &x[3]))
	return ad.Return(ad.Arithmetic(ad.OpDiv, &y, &x[4]))
}`,
		},
		//====================================================
		{`package elemental

import "math"

type Model float64

func pi() float64 {
	return 3.14159
}

func (m Model) Observe(x []float64) float64 {
	y := math.Sin(x[0])
	return y
}`,
			//----------------------------------------------------
			`package elemental

import (
	"math"
	"bitbucket.org/dtolpin/infergo/ad"
)

type Model float64

func pi() float64 {
	return 3.14159
}

func (m Model) Observe(x []float64) float64 {
	ad.Setup(x)
	var y float64
	ad.Assignment(&y, ad.Elemental(math.Sin, &x[0]))
	return ad.Return(&y)
}`,
		},
		//====================================================
		{`package opaque

type Model float64

func pi() float64 {
	return 3.14159
}

func intpow(a float64, n int) float64 {
	pow := 1.
	for i := 0; i != n; i++ {
		pow *= a
	}
	return pow
}

func (m Model) Observe(x []float64) float64 {
	y := pi()
	z := intpow(y, 3)
	return y * z
}`,
			//----------------------------------------------------
			`package opaque

import "bitbucket.org/dtolpin/infergo/ad"

type Model float64

func pi() float64 {
	return 3.14159
}

func intpow(a float64, n int) float64 {
	pow := 1.
	for i := 0; i != n; i++ {
		pow *= a
	}
	return pow
}

func (m Model) Observe(x []float64) float64 {
	ad.Setup(x)
	var y float64
	ad.Assignment(&y, ad.Value(pi()))
	var z float64
	ad.Assignment(&z, ad.Value(intpow(y, 3)))
	return ad.Return(ad.Arithmetic(ad.OpMul, &y, &z))
}`,
		},
		//====================================================
		{`package enter

type Model float64

func (m Model) sum(x, _ float64, y float64) float64 {
	return x + y
}

func (m Model) Observe(x []float64) float64 {
	return m.sum(x[0], x[1], x[2])
}`,
			//----------------------------------------------------
			`package enter

import "bitbucket.org/dtolpin/infergo/ad"

type Model float64

func (m Model) sum(x, _ float64, y float64) float64 {
	ad.Enter(&x, ad.Value(0), &y)
	return ad.Return(ad.Arithmetic(ad.OpAdd, &x, &y))
}

func (m Model) Observe(x []float64) float64 {
	ad.Setup(x)
	return ad.Return(ad.Call(func (_vararg []float64) {
		m.sum(0, 0, 0)
	}, 3, &x[0], &x[1], &x[2]))
}`,
		},
		//====================================================
		{`package call

type Model float64

func (m Model) sum(x, y float64) float64 {
	return x + y
}

func (m Model) Observe(x []float64) float64 {
	return m.sum(x[0], x[1])
}`,
			//----------------------------------------------------
			`package call

import "bitbucket.org/dtolpin/infergo/ad"

type Model float64

func (m Model) sum(x, y float64) float64 {
	ad.Enter(&x, &y)
	return ad.Return(ad.Arithmetic(ad.OpAdd, &x, &y))
}

func (m Model) Observe(x []float64) float64 {
	ad.Setup(x)
	return ad.Return(ad.Call(func (_vararg []float64) {
			m.sum(0, 0)
	}, 2, &x[0], &x[1]))
}`,
		},
		//====================================================
		{`package variadic

type Model float64

func (m Model) sum(x ...float64) float64 {
	return x[0]
}

func (m Model) Observe(x []float64) float64 {
	return m.sum(x[0], x[1], x[2])
}`,
			//----------------------------------------------------
			`package variadic

import "bitbucket.org/dtolpin/infergo/ad"

type Model float64

func (m Model) sum(x ...float64) float64 {
	ad.Enter()
	return ad.Return(&x[0])
}

func (m Model) Observe(x []float64) float64 {
	ad.Setup(x)
	return ad.Return(ad.Call(func (_vararg []float64) {
		m.sum(_vararg...)
	}, 0, &x[0], &x[1], &x[2]))
}`,
		},
		//====================================================
		{`package semivari

type Model float64

func (m Model) sum(x, y ...float64) float64 {
	return x + y[0]
}

func (m Model) Observe(x []float64) float64 {
	return m.sum(x[0], x[1], x[2])
}`,
			//----------------------------------------------------
			`package semivari

import "bitbucket.org/dtolpin/infergo/ad"

type Model float64

func (m Model) sum(x, y ...float64) float64 {
	ad.Enter(&x)
	return ad.Return(ad.Arithmetic(ad.OpAdd, &x, &y[0]))
}

func (m Model) Observe(x []float64) float64 {
	ad.Setup(x)
	return ad.Return(ad.Call(func (_vararg []float64) {
		m.sum(0, _vararg...)
	}, 1, &x[0], &x[1], &x[2]))
}`,
		},
		//====================================================
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
		m.deriv()
		if !equiv(m.pkg.Files["original.go"], c.differentiated) {
			b := new(bytes.Buffer)
			printer.Fprint(b, m.fset, m.pkg.Files["original.go"])

			t.Errorf("model %v:\n---\n%v\n---\n"+
				" not equivalent to \n---\n%v\n---\n",
				m.pkg.Name,
				b.String(),
				c.differentiated)
		}
	}
}

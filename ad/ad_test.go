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
func parseTestModel(sources map[string]string) (
	m *model,
	err error,
) {
	m = &model{prefix: "_"}
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
			return nil, err
		}
	}
	return m, nil
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
	return 0
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
	return 0
}
`,
		},
			map[string]bool{
				"Observe": true,
				"Sample":  true,
			}},
	} {
		m, err := parseTestModel(c.model)
		if err != nil {
			t.Errorf("failed to parse: %s", err)
		}
		err = m.check()
		if err != nil {
			t.Errorf("failed to check %v: %s", m.pkg.Name, err)
		}
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

func TestDesugar(t *testing.T) {
	for _, c := range []struct {
		original, desugared string
	}{
		//====================================================
		{`package define

type Model float64

func (m Model) Observe(x []float64) float64 {
	a, c, e := 0., 'c', 1
	println(c, e)
	return a
}`,
			//----------------------------------------------------
			`package define

type Model float64

func (m Model) Observe(x []float64) float64 {
	var (
		a float64
		c rune
	    e int
	)
	a, c, e = 0., 'c', 1
	println(c, e)
	return a
}`,
		},
		//====================================================
		{`package definesome

type Model float64

func (m Model) Observe(x []float64) float64 {
	err := "success"
	a, err := 0, "failure"
	println(a, err)
	return 1
}`,
			//----------------------------------------------------
			`package definesome

type Model float64

func (m Model) Observe(x []float64) float64 {
	var err string
	err = "success"
	var a int
	a, err = 0, "failure"
	println(a, err)
	return 1
}`,
		},
		//====================================================
		{`package self
type Model float64

type Foo struct {y float64}

func (m Model) Observe(x []float64) float64 {
	foo := new(Foo)
	return foo.y
}`,
			//----------------------------------------------------
			`package self
type Model float64

type Foo struct {y float64}

func (m Model) Observe(x []float64) float64 {
	var foo *Foo
	foo = new(Foo)
	return foo.y
}`,
		},
		//====================================================
		{`package qualified
import "go/ast"
import tok "go/token"

type Model float64

func (m Model) Observe(x []float64) float64 {
	id := &ast.Ident{}
	pos := &tok.Position{}
	println(id, pos)
	return 0
}`,
			//----------------------------------------------------
			`package qualified
import "go/ast"
import tok "go/token"

type Model float64

func (m Model) Observe(x []float64) float64 {
	var id *ast.Ident
	id = &ast.Ident{}
	var pos *tok.Position
	pos = &tok.Position{}
	println(id, pos)
	return 0
}`,
		},
		//====================================================
		{`package unimported
import "go/ast"

type Model float64

func (m Model) Observe(x []float64) float64 {
	pos := ast.Ident{}.NamePos
	pos = pos
	return 0
}`,
			//----------------------------------------------------
			`package unimported
import (
	"go/ast"
	"go/token"
)

type Model float64

func (m Model) Observe(x []float64) float64 {
	var pos token.Pos
	pos = ast.Ident{}.NamePos
	pos = pos
	return 0
}`,
		},
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
	return 0
}`,
			//----------------------------------------------------
			`package simplestmt

type Model float64

func foo() (int, interface{}) {return 0, &struct {}{}}

func (m Model) Observe(x []float64) float64 {
	if _, err := foo(); err != nil {
		println("error")
	}
	return 0
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
		m, err := parseTestModel(map[string]string{
			"original.go": c.original,
		})
		if err != nil {
			t.Errorf("failed to parse: %s", err)
		}
		err = m.check()
		if err != nil {
			t.Errorf("failed to check %v: %s", m.pkg.Name, err)
		}
		methods, err := m.collectMethods()
		for _, method := range methods {
			m.desugar(method)
		}
		if !equiv(m.pkg.Files["original.go"], c.desugared) {
			b := new(bytes.Buffer)
			printer.Fprint(b, m.fset, m.pkg.Files["original.go"])

			t.Errorf("model %v:\n---\n%v\n---\n"+
				" not equivalent to \n---\n%v\n---\n",
				m.pkg.Name,
				b.String(),
				c.desugared)
		}
	}
}

func TestRewrite(t *testing.T) {
	for _, c := range []struct {
		original, differentiated string
	}{
		//====================================================
		{`package lit

type Model float64

func (m Model) Observe(x []float64) float64 {
	return 0.5
}

func (m Model) Count() int {
	return 0
}`,
			//----------------------------------------------------
			`package lit

import "bitbucket.org/dtolpin/infergo/ad"

type Model float64

func (m Model) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	return ad.Return(ad.Value(0.5))
}

func (m Model) Count() int {
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
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	var y float64
	ad.Assignment(&y, ad.Value(1.))
	return ad.Return(&y)
}

func (m Model) Count() int {
	y := 1
	return y
}`,
		},
		//====================================================
		{`package constant

import "math"

type Model float64

const foo = 1

func (m Model) Observe(x []float64) float64 {
	return foo
}

func (m Model) Pi () float64 {
	return math.Pi
}`,
			//----------------------------------------------------
			`package constant

import (
	"math"
	"bitbucket.org/dtolpin/infergo/ad"
)

type Model float64

const foo = 1

func (m Model) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	return ad.Return(ad.Value(foo))
}

func (m Model) Pi () float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		panic("Pi called outside Observe")
	}
	return ad.Return(ad.Value(math.Pi))
}`,
		},
		//====================================================
		{`package assign

type Model float64

func (m Model) Observe(x []float64) float64 {
	y := 1.
	z, _ := 2., y
	_ = y
	return z
}`,
			//----------------------------------------------------
			`package assign

import "bitbucket.org/dtolpin/infergo/ad"

type Model float64

func (m Model) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	var y float64
	ad.Assignment(&y, ad.Value(1.))
	var z float64
	ad.ParallelAssignment(&z, ad.Value(0), ad.Value(2.), &y)
	ad.Assignment(ad.Value(0), &y)
	return ad.Return(&z)
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
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	return ad.Return(&x[0])
}`,
		},
		//====================================================
		{`package mapentry

type Model float64

func (m Model) Observe(x []float64) float64 {
	y := make(map[string]float64)
	y["a"] = 1
	return y["a"]
}`,
			//----------------------------------------------------
			`package mapentry

import "bitbucket.org/dtolpin/infergo/ad"

type Model float64

func (m Model) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	var y map[string]float64
	y = make(map[string]float64)
	y["a"] = 1
	return ad.Return(ad.Value(y["a"]))
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
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
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
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
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
	y = -x[0]
	return y
}`,
			//----------------------------------------------------
			`package unary

import "bitbucket.org/dtolpin/infergo/ad"

type Model float64

func (m Model) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	var y float64
	ad.Assignment(&y, &x[0])
	ad.Assignment(&y, ad.Arithmetic(ad.OpNeg, &x[0]))
	return ad.Return(&y)
}`,
		},
		//====================================================
		{`package binary

import "math"

type Model float64

const one = 1

func (m Model) Observe(x []float64) float64 {
	y := x[0] + x[1]
	y = y + 1
	y = y - x[2]
	y = y * x[3]
	y = y + one       // local named constant
	y = y + math.Pi   // imported named constant
	return y / x[4]
}`,
			//----------------------------------------------------
			`package binary

import (
	"math"
	"bitbucket.org/dtolpin/infergo/ad"
)

type Model float64

const one = 1

func (m Model) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	var y float64
	ad.Assignment(&y, ad.Arithmetic(ad.OpAdd, &x[0], &x[1]))
	ad.Assignment(&y, ad.Arithmetic(ad.OpAdd, &y, ad.Value(1)))
	ad.Assignment(&y, ad.Arithmetic(ad.OpSub, &y, &x[2]))
	ad.Assignment(&y, ad.Arithmetic(ad.OpMul, &y, &x[3]))
	ad.Assignment(&y, ad.Arithmetic(ad.OpAdd, &y, ad.Value(one)))
	ad.Assignment(&y, ad.Arithmetic(ad.OpAdd, &y, ad.Value(math.Pi)))
	return ad.Return(ad.Arithmetic(ad.OpDiv, &y, &x[4]))
}`,
		},
		//====================================================
		{`package folding

import "math"

type Model float64

func (m Model) Observe(_ []float64) float64 {
	y := -0.5
	y = 1 + 2
	y = math.Pi - 1
	return y
}`,
			//----------------------------------------------------
			`package folding

import (
	"math"
	"bitbucket.org/dtolpin/infergo/ad"
)

type Model float64

func (m Model) Observe(_ []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup([]float64{})
	}
	var y float64
	ad.Assignment(&y, ad.Value(-0.5))
	ad.Assignment(&y, ad.Value(3))
	ad.Assignment(&y, ad.Value(2.141592653589793))
	return ad.Return(&y)
}`,
		},
		//====================================================
		{`package elemental

import "math"

type Model float64

func first(x []float64) float64 {
	return x[0]
}

func (m Model) Observe(x []float64) float64 {
	y := math.Sin(x[0])
	z := first(x)
	return y + z
}`,
			//----------------------------------------------------
			`package elemental

import (
	"math"
	"bitbucket.org/dtolpin/infergo/ad"
)

type Model float64

func first(x []float64) float64 {
	return x[0]
}

func (m Model) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	var y float64
	ad.Assignment(&y, ad.Elemental(math.Sin, &x[0]))
	var z float64
	ad.Assignment(&z, ad.Vlemental(first, x))
	return ad.Return(ad.Arithmetic(ad.OpAdd, &y, &z))
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
	z = float64(z)
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
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	var y float64
	ad.Assignment(&y, ad.Value(pi()))
	var z float64
	ad.Assignment(&z, ad.Value(intpow(y, 3)))
	ad.Assignment(&z, ad.Value(float64(z)))
	return ad.Return(ad.Arithmetic(ad.OpMul, &y, &z))
}`,
		},
		//====================================================
		{`package opaque_arg

type Model float64

func count() int {
	return 0
}

func (m Model) IntPow(a float64, n int) float64 {
	pow := 1.
	for i := 0; i != n; i++ {
		pow *= a
	}
	return pow
}

func (m Model) Observe(x []float64) float64 {
	return m.IntPow(x[0], count())
}`,
			//----------------------------------------------------
			`package opaque_arg

import "bitbucket.org/dtolpin/infergo/ad"

type Model float64

func count() int {
	return 0
}

func (m Model) IntPow(a float64, n int) float64 {
	if ad.Called() {
		ad.Enter(&a)
	} else {
		panic("IntPow called outside Observe")
	}
	var pow float64
	ad.Assignment(&pow, ad.Value(1.))
	for i := 0; i != n; i = i + 1 {
		ad.Assignment(&pow, ad.Arithmetic(ad.OpMul, &pow, &a))
	}
	return ad.Return(&pow)
}

func (m Model) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	return ad.Return(ad.Call(func(_ []float64) {
		m.IntPow(0, count())
	}, 1, &x[0]))
}`,
		},
		//====================================================
		{`package enter

type Model float64

func (m Model) sum(x, _ float64, y float64) float64 {
	return x + y
}

func (m Model) z(x []float64, y float64) float64 {
	return y
}

func (m Model) Observe(x []float64) float64 {
	return m.sum(x[0], x[1], x[2]) - m.z(x[:2], x[2])
}`,
			//----------------------------------------------------
			`package enter

import "bitbucket.org/dtolpin/infergo/ad"

type Model float64

func (m Model) sum(x, _ float64, y float64) float64 {
	if ad.Called() {
		ad.Enter(&x, ad.Value(0), &y)
	} else {
		panic("sum called outside Observe")
	}
	return ad.Return(ad.Arithmetic(ad.OpAdd, &x, &y))
}

func (m Model) z(x []float64, y float64) float64 {
	if ad.Called() {
		ad.Enter(&y)
	} else {
		panic("z called outside Observe")
	}
	return ad.Return(&y)
}

func (m Model) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	return ad.Return(ad.Arithmetic(ad.OpSub,
		ad.Call(func(_ []float64) {
			m.sum(0, 0, 0)
		}, 3, &x[0], &x[1], &x[2]),
		ad.Call(func(_ []float64) {
			m.z(x[:2], 0)
		}, 1, &x[2])))
}`,
		},
		//====================================================
		{`package multival

type Model float64

func (m Model) vals(x float64, y float64) (float64, float64) {
	return x, y
}

func (m Model) Observe(x []float64) float64 {
	y, z := m.vals(x[0], x[1])
	return y + z
}`,
			//----------------------------------------------------
			`package multival

import "bitbucket.org/dtolpin/infergo/ad"

type Model float64

func (m Model) vals(x float64, y float64) (float64, float64) {
	return x, y
}

func (m Model) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	var (
		y	float64
		z	float64
	)

	y, z = m.vals(x[0], x[1])
	return ad.Return(ad.Arithmetic(ad.OpAdd, &y, &z))
}`,
		},
		//====================================================
		{`package importad
import "bitbucket.org/dtolpin/infergo/dist/ad"

type Model float64

func (m Model) Observe(x []float64) float64 {
	return dist.Normal.Observe(x)
}`,
			//----------------------------------------------------
			`package importad

import (
	"bitbucket.org/dtolpin/infergo/dist/ad"
	"bitbucket.org/dtolpin/infergo/ad"
)

type Model float64

func (m Model) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	return ad.Return(ad.Call(func (_ []float64) {
			dist.Normal.Observe(x)
	}, 0))
}`,
		},
		//====================================================
		{`package importautoad
import (
	"bitbucket.org/dtolpin/infergo/dist"
	// "ad/" is added to the first import of each package, 
	// the remaining import (with different names) are 
	// left unmodified and can be used to access
	// undifferentiated versions of the package
	ud "bitbucket.org/dtolpin/infergo/dist"
)

type Model float64

func foo() float64 {
	// Normal is used in undifferentiated context
	return ud.Normal.Logp(0, 1, 0)
}

func (m Model) Observe(x []float64) float64 {
	return dist.Normal.Observe(x)
}`,
			//----------------------------------------------------
			`package importautoad

import (
	"bitbucket.org/dtolpin/infergo/dist/ad"
	"bitbucket.org/dtolpin/infergo/ad"
	ud "bitbucket.org/dtolpin/infergo/dist"
)

type Model float64

func foo() float64 {
	return ud.Normal.Logp(0, 1, 0)
}

func (m Model) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	return ad.Return(ad.Call(func (_ []float64) {
			dist.Normal.Observe(x)
	}, 0))
}`,
		},
		//====================================================
		{`package variadic

type Model float64

func (m Model) sum(x ...float64) float64 {
	return x[0]
}

func (m Model) Sum(x ...float64) float64 {
	return m.sum(x...)
}

func (m Model) Observe(x []float64) float64 {
	return m.sum(x[0], x[1], x[2])
}`,
			//----------------------------------------------------
			`package variadic

import "bitbucket.org/dtolpin/infergo/ad"

type Model float64

func (m Model) sum(x ...float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		panic("sum called outside Observe")
	}
	return ad.Return(&x[0])
}

func (m Model) Sum(x ...float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		panic("Sum called outside Observe")
	}
	return ad.Return(ad.Call(func (_ []float64) {
		m.sum(x...)
	}, 0))
}

func (m Model) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	return ad.Return(ad.Call(func (_vararg []float64) {
		m.sum(_vararg...)
	}, 0, &x[0], &x[1], &x[2]))
}`,
		},
		//====================================================
		{`package semivari

type Model float64

func (m Model) sum(x float64, y ...float64) float64 {
	return x + y[0]
}

func (m Model) Observe(x []float64) float64 {
	return m.sum(x[0], x[1], x[2])
}`,
			//----------------------------------------------------
			`package semivari

import "bitbucket.org/dtolpin/infergo/ad"

type Model float64

func (m Model) sum(x float64, y ...float64) float64 {
	if ad.Called() {
		ad.Enter(&x)
	} else {
		panic("sum called outside Observe")
	}
	return ad.Return(ad.Arithmetic(ad.OpAdd, &x, &y[0]))
}

func (m Model) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	return ad.Return(ad.Call(func (_vararg []float64) {
		m.sum(0, _vararg...)
	}, 1, &x[0], &x[1], &x[2]))
}`,
		},
		//====================================================
		{`package composite

type Model float64

func (m Model) Observe(x []float64) float64 {
	
	return m.Observe([]float64{x[0]})
}`,
			//----------------------------------------------------
			`package composite

import "bitbucket.org/dtolpin/infergo/ad"

type Model float64

func (m Model) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	return ad.Return(ad.Call(func (_ []float64) {
		m.Observe([]float64{x[0]})
	}, 0))
		//====================================================
}`,
		},
	} {
		m, err := parseTestModel(map[string]string{
			"original.go": c.original,
		})
		if err != nil {
			t.Errorf("failed to parse: %s", err)
		}
		err = m.check()
		if err != nil {
			t.Errorf("failed to check %v: %s", m.pkg.Name, err)
		}
		err = m.deriv()
		if err != nil {
			t.Errorf("failed to differentiate %v: %s", m.pkg.Name, err)
		}
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

func TestDerivErrors(t *testing.T) {
	for _, c := range []struct {
		erroneous string
		err       string
	}{
		{
			`package pkgname
import (
	"go/ast"
	token "math" // take the name token need to name token.Pos type
)

type Model float64

func (m Model) Observe(x []float64) float64 {
	pos := ast.Ident{}.NamePos
	pos = pos
	return token.Sqrt(1)
}
`,
			"erroneous.go:10:2: cannot name package \"go/token\": " +
				"not imported and name \"token\" is taken",
		},
		{
			`package adoverride
import ad "fmt"

type Model float64

func (m Model) Observe(x []float64) float64 {
	ad.Println()
	return 0
}
`,
			"erroneous.go:2:8: package name \"ad\" is reserved",
		},
		{
			`package reserved

type Model float64

func (m Model) Observe(x []float64) float64 {
	_y := 1.
	return x[0] + _y
}
`,
			"erroneous.go:6:2: identifier \"_y\" is reserved",
		},
	} {
		m, err := parseTestModel(map[string]string{
			"erroneous.go": c.erroneous,
		})
		if err != nil {
			t.Errorf("failed to parse: %s", err)
			continue
		}
		err = m.check()
		if err != nil {
			t.Errorf("failed to check %v: %s", m.pkg.Name, err)
		}
		err = m.deriv()
		if err == nil {
			t.Fatalf("should fail on %v: %s", m.pkg.Name, c.err)
		}
		if err.Error() != c.err {
			t.Errorf("wrong error on %v: got %q, want %q",
				m.pkg.Name, err, c.err)
		}
	}
}

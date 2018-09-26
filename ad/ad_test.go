package ad

import (
	"fmt"
	"bytes"
	"go/token"
	// "go/ast"
	"go/parser"
	"go/printer"
	"testing"
)

// Tooling for comparing models

// wrapper around model body
func srcModel(body string) string {
	return fmt.Sprintf(`package mymodel

func (m *interface{}) Observe (parameters []float64) float64 {
%s
}`, body)
}

// wrapped around differentiated model body
func diffModel(body string) string {
	return fmt.Sprintf(`package mymodel

func (m *interface{}) Observe (parameters []float64) (float64, []float64) {
%s
}`, body)
}

func sourcesEqual(got, expected string) bool {
	// normalize source code by parsing and printing
	fileSet := token.NewFileSet()

	// parse it
	gotTree, error := parser.ParseFile(fileSet, "", got, 0)
	if(error != nil) {
		panic(error)
	}
	expectedTree, error := parser.ParseFile(fileSet, "", expected, 0)
	if(error != nil) {
		panic(error)
	}

	// print it
	gotBuffer := new(bytes.Buffer)
	expectedBuffer := new(bytes.Buffer)
	printer.Fprint(gotBuffer, fileSet, gotTree)
	printer.Fprint(expectedBuffer, fileSet, expectedTree)

	println(gotTree, expectedTree)

	// compare strings
	return gotBuffer.String() == expectedBuffer.String()
}

// check that templates work and parsing normalizes
func TestSourcesEqual(t *testing.T) {
	for _, m := range []func (string) string {srcModel, diffModel} {
		for _, c := range []struct {
			got, expected string
			equal bool
		}{
			{"return 0.", "return 0.", true},
			{"return 0.", "return    0.", true},
			{"return 0.", "return 1.", false},
			{"return 0.", "return 0", false},
		} {
			g := m(c.got)
			e := m(c.expected)
			if sourcesEqual(g, e) != c.equal {
				t.Errorf("'%s' and '%s' should %sbe equal",
					g, e,
					map[bool]string {
						false: "not ",
						true: "",
					}[c.equal])
			}
		}
	}
}

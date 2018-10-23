// Automatic differentiation of a model
package ad

// A model is defined in it's own package. The model must
// implement interface model.Model. In the model's source code:
//   1. Method Observe of interface model.Model is
//      differentiated.
//   2. All methods on the type implementing model.Model
//      are differentiated.
//   3. Within the methods, the following is differentiated:
//      a) assignments to float64;
//      b) returns of float64;
//      c) standalone calls to methods on the type implementing
//         model.Model (apparently called for side  effects on
//         the model).
//
// Functions are considered elementals (and must have a
// registered derivative):
//   a) if they are defined in a different package, and
//   b) their signature is of kind
//                  func (float64, ...float64) float64
//      that is, at least one float64 argument and float64 return
//      value. For example, function
//                  func (float64, float64, float64) float64
//      is considered elemental, while functions
//                  func ([]float64) float64
//                  func (int, float64) float64
//      are not.
//
// Derivatives do not propagate through a function that is not
// an elemental or a call to a model method. If a derivative is
// not registered for an elemental, calling the elemental in a
// differentiated context will cause a run-time error.
//
// The differentiated model is put into subpackage 'ad'
// of the model's package.

import (
	"bufio"
	"fmt"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"os"
	"path"
	"strings"
)

// Deriv differentiates a model. The original model is
// in the package located at model. The differentiated model
// is written to model/ad.
func Deriv(model string) error {
	// Read the source code.
	// If there are any errors in the source code, stop.
	pkg, err := parsePackage(model)
	if err != nil {
		return err
	}

	// Typecheck the package.

	// Rewrite the AST to add automatic differentation.

	// Write the source code to the updated package.
	return writePackage(path.Join(model, "ad"), pkg)
}

// parsePackage parses the model's source code and returns
// the parsed package and an error. If the model was parsed
// successfully, the error is nil.
func parsePackage(model string) (pkg *ast.Package, err error) {
	// Read the source code.
	// If there are any errors in the source code, stop.
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, model, nil, 0)
	if err != nil { // parse error
		goto End
	}

	// There should be a single package, retrieve it.
	// If there is more than a single package, stop.
	for k, v := range pkgs {
		if pkg != nil {
			err = fmt.Errorf("multiple packages in %s: %s and %s",
				model, pkg.Name, k)
			goto End
		}
		pkg = v
	}

End:
	return pkg, err
}

// writePackage writes the differentiated model as
// a Go package source.
func writePackage(adModel string, pkg *ast.Package) error {
	// Create the directory for the differentiated model.
	err := os.Mkdir(adModel, os.ModePerm)
	if err != nil &&
		// The only error we can tolerate is that the directory
		// already exists (for example from an earlier
		// differentiation).
		!strings.Contains(err.Error(), "file exists") {
		return err
	}

	// Write files to the ad subpackage under the same names.
	// A fresh file set is created because it is mainly useful
	// for comments, and we dropped the comments anyway.
	fset := token.NewFileSet()
	for fPath, fAst := range pkg.Files {
		_, fName := path.Split(fPath)
		f, err := os.Create(path.Join(adModel, fName))
		defer f.Close()
		if err != nil {
			return err
		}
		w := bufio.NewWriter(f)
		defer w.Flush()
		printer.Fprint(w, fset, fAst)
	}

	return nil
}

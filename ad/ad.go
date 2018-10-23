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
    "strings"
    "os"
    "path"
    "fmt"
    "bufio"
    "go/token"
    "go/ast"
    "go/parser"
    "go/printer"
)

// Differentiate differentiates a model. The original model is
// in the package located at model. The differentiated model
// is written to model/ad.
func Differentiate(model string) error {
    // Read the source code.
    // If there are any errors in the source code, stop.
    fset := token.NewFileSet()
    pkgs, err := parser.ParseDir(fset, model, nil, 0)
    if(err != nil) {
        return err
    }
    // there should be a single package, retrieve it
    var pkg *ast.Package
    for k, v := range pkgs {
        if pkg != nil {
            return fmt.Errorf("multiple packages: %s and %s",
                pkg.Name, k)
        }
        pkg = v
    }

    // Typecheck the package.

    // Rewrite the AST to add automatic differentation.

    // Write the source code to the updated package.
    
    adPath := path.Join(model, "ad")
    err = os.Mkdir(adPath, os.ModePerm)
    if err != nil && 
        !strings.Contains(err.Error(), "file exists") {
        return err
    }

    for fPath, fAst := range pkg.Files {
        _, fName := path.Split(fPath)
        f, err := os.Create(path.Join(adPath, fName))
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

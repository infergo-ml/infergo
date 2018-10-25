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
	"go/importer"
	"go/parser"
	"go/printer"
	"go/token"
	"go/types"
	"golang.org/x/tools/go/ast/astutil"
	"os"
	"path"
	"strings"
)

// model contains shared data structures for compiled
type model struct {
	fset *token.FileSet
	pkg  *ast.Package
	info *types.Info
}

// Deriv differentiates a model. The original model is
// in the package located at modelPath. The differentiated model
// is written to modelPath/ad.
func Deriv(modelPath string) (err error) {
	// Read the model.
	m := &model{}
	err = parseModel(m, modelPath)
	if err != nil {
		return err
	}

	// Typecheck the model to build the info structure.
	err = checkModel(m, modelPath)
	if err != nil {
		return err
	}

	// Differentiate the model through rewriting the ASTs
	// in place.
	err = derivModel(m)
	if err != nil {
		return err
	}

	// Finally write the model to subpackage 'ad'.
	err = writeModel(m, path.Join(modelPath, "ad"))

	return err
}

// Parsing

// parseModel parses the model's source code and returns
// the parsed package and an error. If the model was parsed
// successfully, the error is nil.
func parseModel(m *model, modelPath string) (err error) {
	// Read the source code.
	// If there are any errors in the source code, stop.
	m.fset = token.NewFileSet()
	pkgs, err := parser.ParseDir(m.fset, modelPath, nil, 0)
	if err != nil { // parse error
		goto End
	}

	// There should be a single package, retrieve it.
	// If there is more than a single package, stop.
	for k, v := range pkgs {
		if m.pkg != nil {
			err = fmt.Errorf("multiple packages in %q: %s and %s",
				modelPath, m.pkg.Name, k)
			goto End
		}
		m.pkg = v
	}

End:
	return err
}

// checkModel typechecks the model and builds the info
// structure.
func checkModel(m *model, modelPath string) (err error) {
	conf := types.Config{Importer: importer.Default()}
	// Check expects the package as a slice of file ASTs.
	var files []*ast.File
	for _, file := range m.pkg.Files {
		files = append(files, file)
	}
	m.info = &types.Info{
		Defs: make(map[*ast.Ident]types.Object),
		Uses: make(map[*ast.Ident]types.Object),
	}
	_, err = conf.Check(modelPath, m.fset, files, m.info)
	return err
}

// Differentiation

const infergoImport = "bitbucket.org/dtolpin/infergo/ad"

// derivModel differentiates the model through rewriting
// the ASTs.
func derivModel(m *model) (err error) {
	modelTypes, err := collectModelTypes(m)
	if err != nil {
		return err
	}

	// Add infergo import to each file with model methods
	modelFiles, err := collectModelFiles(m, modelTypes)
	if err != nil {
		return err
	}
	for _, file := range modelFiles {
		astutil.AddImport(m.fset, file, infergoImport)
	}

	// Differentiate each model method
	modelMethods, err := collectModelMethods(m, modelTypes)
	if err != nil {
		return err
	}
	for _, method := range modelMethods {
		if strings.Compare(method.Name.Name, "Observe") == 0 {
			// Differentiate the main model method.
		} else {
			// Differentiate a model method which
			// may be called from Observe.
		}
	}

	return err
}

// collectModelTypes collects and returns the types of models
// defined in the package.
func collectModelTypes(m *model) (modelTypes []types.Type, err error) {
	// Identify the model type (or types)
	modelTypes = make([]types.Type, 0, 1)
	for _, file := range m.pkg.Files {
		for _, d := range file.Decls {
			if d, ok := d.(*ast.FuncDecl); ok {
				if strings.Compare(d.Name.Name, "Observe") == 0 {
					// May be the observe method, but check the
					// signature.
					t := m.info.TypeOf(d.Name).(*types.Signature)
					if isObserveSignature(t) {
						modelTypes = append(modelTypes,
							t.Recv().Type())
					}
				}
			}
		}
	}
	if len(modelTypes) == 0 {
		err = fmt.Errorf("no model in package %s", m.pkg.Name)
	}
	return modelTypes, err
}

// isObserveSignature returns true iff the signature is that of
// the Observe method: func ([]float64) float64
func isObserveSignature(t *types.Signature) (ok bool) {
	// Consider pattern matching for go/types
	ok = true

	// Is a method
	ok = ok && t.Recv() != nil

	// Returns a single float64
	ok = ok && t.Results().Len() == 1 // returns a single result
	if !ok {
		return ok
	}
	rt, ok := t.Results().At(0).Type().(*types.Basic)
	if !ok {
		return ok
	}
	ok = rt.Kind() == types.Float64 // the result is float64

	// Accepts a single []float64
	ok = ok && t.Params().Len() == 1 // accepts a single parameter
	if !ok {
		return ok
	}
	pt, ok := t.Params().At(0).Type().(*types.Slice) // a slice
	if !ok {
		return ok
	}
	et, ok := pt.Elem().(*types.Basic)
	if !ok {
		return ok
	}
	ok = et.Kind() == types.Float64 // the element type is float64

	return ok
}

// collectModelFiles collects ASTs of files
// where the model methods appear so that
// imports can be added.
func collectModelFiles(m *model, modelTypes []types.Type) (
	modelFiles []*ast.File,
	err error,
) {
	for _, file := range m.pkg.Files {
		for _, d := range file.Decls {
			if d, ok := d.(*ast.FuncDecl); ok &&
				isModelMethod(m, modelTypes, d) {
				modelFiles = append(modelFiles, file)
				break
			}
		}
	}

	return modelFiles, err
}

// collectModelMethods collects ASTs of methods
// defined on the models.
func collectModelMethods(m *model, modelTypes []types.Type) (
	modelMethods []*ast.FuncDecl,
	err error,
) {
	// We will mostly have a single model type; a linear
	// lookup is the way to go (see isModelType below).
	for _, file := range m.pkg.Files {
		for _, d := range file.Decls {
			if d, ok := d.(*ast.FuncDecl); ok &&
				isModelMethod(m, modelTypes, d) {
				modelMethods = append(modelMethods, d)
			}
		}
	}

	return modelMethods, err
}

// isModelMethod returns true iff the function declaration
// is a model method.
func isModelMethod(
	m *model,
	modelTypes []types.Type,
	d *ast.FuncDecl,
) bool {
	if d.Recv == nil {
		return false
	}
	t := m.info.TypeOf(d.Name).(*types.Signature)
	return isModelType(modelTypes, t.Recv().Type())
}

// isModelType returns true iff the type is a model type;
// a type is a model type if the method on this type is
// a model method.
func isModelType(modelTypes []types.Type, t types.Type) bool {
	for _, mt := range modelTypes {
		if types.Identical(mt, t) {
			return true
		}
		if mt, ok := mt.(*types.Pointer); ok &&
			types.Identical(mt.Elem(), t) {
			return true
		}
	}
	return false
}

// Writing

// writeModel writes the differentiated model as
// a Go package source.
func writeModel(m *model, adModelPath string) (err error) {
	// Create the directory for the differentiated model.
	err = os.Mkdir(adModelPath, os.ModePerm)
	if err != nil &&
		// The only error we can tolerate is that the directory
		// already exists (for example from an earlier
		// differentiation).
		!strings.Contains(err.Error(), "file exists") {
		return err
	}
	err = nil

	// Write files to the ad subpackage under the same names.
	for fpath, file := range m.pkg.Files {
		_, fname := path.Split(fpath)
		f, err := os.Create(path.Join(adModelPath, fname))
		defer f.Close()
		if err != nil {
			return err
		}
		w := bufio.NewWriter(f)
		defer w.Flush()
		printer.Fprint(w, m.fset, file)
	}

	return err
}

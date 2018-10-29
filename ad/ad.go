// Automatic differentiation of a model
package ad

// A model is defined in it's own package. The model must
// implement interface model.Model. In the model's source code:
//   1. Method Observe of interface model.Model is
//      differentiated.
//   2. All methods on the type implementing model.Model
//      are differentiated.
//   3. Within the methods, the following is differentiated:
//      a) assignments to float64 (including parallel
//         assignments if all values are of type float64);
//      b) returns of float64;
//      c) standalone calls to methods on the type implementing
//         model.Model (apparently called for side  effects on
//         the model).
//   4. Non-dummy dentifiers starting with '_' are reserved.
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

// Structure model contains shared data structures for
// differentiating the model. Functions operating on *model are
// defined as method to use shorter names.
type model struct {
	path string
	fset *token.FileSet
	pkg  *ast.Package
	info *types.Info
	typs []types.Type
}

// Deriv differentiates a model. The original model is in the
// package located at mpath. The differentiated model is written
// to mpath/ad.
func Deriv(mpath string) (err error) {
	// Read the model.
	m := &model{}
	err = m.parse(mpath)
	if err != nil {
		return err
	}

	// Typecheck the model to build the info structure.
	err = m.check()
	if err != nil {
		return err
	}

	// Differentiate the model through rewriting the ASTs
	// in place.
	err = m.deriv()
	if err != nil {
		return err
	}

	// Finally write the model.
	err = m.write()

	return err
}

// Parsing

// parse parses the model's source code and returns the parsed
// package and an error. If the model was parsed successfully,
// the error is nil.
func (m *model) parse(mpath string) (err error) {
	// Read the source code.
	// If there are any errors in the source code, stop.
	m.path = mpath
	m.fset = token.NewFileSet()
	pkgs, err := parser.ParseDir(m.fset, m.path, nil, 0)
	if err != nil { // parse error
		goto End
	}

	// There should be a single package, retrieve it.
	// If there is more than a single package, stop.
	for k, v := range pkgs {
		if m.pkg != nil {
			err = fmt.Errorf("multiple packages in %q: %s and %s",
				m.path, m.pkg.Name, k)
			goto End
		}
		m.pkg = v
	}

End:
	return err
}

// check typechecks the model and builds the info structure.
func (m *model) check() (err error) {
	conf := types.Config{Importer: importer.Default()}
	// Check expects the package as a slice of file ASTs.
	var files []*ast.File
	for _, file := range m.pkg.Files {
		files = append(files, file)
	}
	m.info = &types.Info{
		Defs:  make(map[*ast.Ident]types.Object),
		Uses:  make(map[*ast.Ident]types.Object),
		Types: make(map[ast.Expr]types.TypeAndValue),
        Selections: make(map[*ast.SelectorExpr]*types.Selection),
	}
	_, err = conf.Check(m.path, m.fset, files, m.info)
	return err
}

// Differentiation

const infergoImport = "bitbucket.org/dtolpin/infergo/ad"

// deriv differentiates the model through rewriting the ASTs.
func (m *model) deriv() (err error) {
	if err = m.collectTypes(); err != nil {
		return err
	}

	// Differentiate each model method

	methods, err := m.collectMethods()
	if err != nil {
		return err
	}

	// Simplify the code first so that differentiation
	// is less cumbersome.
	for _, method := range methods {
		err = m.simplify(method)
		if err != nil {
			return err
		}
	}

	// Finally, rewrite the code using tape-writing calls.
	for _, method := range methods {
		// Add the import (safe to add more than once)
		fname := m.fset.Position(method.Pos()).Filename
		astutil.AddImport(m.fset, m.pkg.Files[fname], infergoImport)

		err = m.rewrite(method)
		if err != nil {
			return err
		}
	}

	return err
}

// collectTypes collects and returns the types of models defined
// in the package.
func (m *model) collectTypes() (err error) {
	// Identify the model type (or types)
	m.typs = make([]types.Type, 0, 1)
	for _, file := range m.pkg.Files {
		for _, d := range file.Decls {
			if d, ok := d.(*ast.FuncDecl); ok {
				if strings.Compare(d.Name.Name, "Observe") == 0 {
					// May be the observe method, but check the
					// signature.
					t := m.info.TypeOf(d.Name).(*types.Signature)
					if isObserve(t) {
						m.typs = append(m.typs, t.Recv().Type())
					}
				}
			}
		}
	}
	if len(m.typs) == 0 {
		err = fmt.Errorf("no model in package %s", m.pkg.Name)
	}
	return err
}

// isObserve returns true iff the signature is that of the
// Observe method: func ([]float64) float64
func isObserve(t *types.Signature) (ok bool) {
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
	ok = ok && t.Params().Len() == 1
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

// collectMethods collects ASTs of methods defined on the
// models.
func (m *model) collectMethods() (
	methods []*ast.FuncDecl,
	err error,
) {
	// We will mostly have a single model type; a linear
	// lookup is the way to go (see iaType below).
	for _, file := range m.pkg.Files {
		for _, d := range file.Decls {
			if d, ok := d.(*ast.FuncDecl); ok &&
				m.isMethod(d) {
				methods = append(methods, d)
			}
		}
	}

	return methods, err
}

// isMethod returns true iff the function declaration is a model
// method.
func (m *model) isMethod(
	d *ast.FuncDecl,
) bool {
	if d.Recv == nil {
		return false
	}
	t := m.info.TypeOf(d.Name).(*types.Signature)
	return m.isType(t.Recv().Type())
}

// isType returns true iff typ is one of model types, or
// pointed to by one of typs.
func (m *model) isType(typ types.Type) bool {
	for _, t := range m.typs {
		if types.Identical(t, typ) {
			return true
		}
		if t, ok := t.(*types.Pointer); ok &&
			types.Identical(t.Elem(), typ) {
			return true
		}
	}
	return false
}

// errOnPanic turns panic from astutil.Apply into an error,
// for consistent diagnostics.
func errOnPanic(
    caller string, 
    err *error, 
    pos token.Position,
) func() {
    return func() {
        if r := recover(); r != nil {
            *err = fmt.Errorf("%v:%d:%d: %v: %v",
            pos.Filename, pos.Line, pos.Column, caller, r)
        }
    }
}

// simplify rewrites the syntax tree of a method to
// differentiate and desugars the syntax, to make the
// differentiate code simpler to write and debug.
func (m *model) simplify(method *ast.FuncDecl) (err error) {
	// Apply panics on errors. When Apply panics, we return the
	// error as do other functions.
	defer errOnPanic(
        "simplify",
        &err,
        m.fset.Position(method.Pos()),
    )()

	astutil.Apply(method,
		func(c *astutil.Cursor) bool {
			n := c.Node()
            if n != nil && n.Pos() != token.NoPos {
                defer errOnPanic(
                    "simplify/pre",
                    &err,
                    m.fset.Position(n.Pos()),
                )()
            }
			switch n := n.(type) {
			case *ast.AssignStmt:
				switch n.Tok {
				case token.ASSIGN:
					// Do nothing, all is well.
				case token.DEFINE:
					// Split into declaration and assignment.

					// Declaration.
					for i := 0; i != len(n.Lhs); i++ {
						ident := n.Lhs[i].(*ast.Ident)
						// The shortest way from go/types to go/ast
						// is to stringify and reparse.
						typ := m.info.TypeOf(n.Lhs[i])
						typast, err := parser.ParseExpr(typ.String())
						if err != nil {
							panic(fmt.Sprintf(
								"cannot parse type %v: %v", typ, err))
						}
						spec := &ast.ValueSpec{
							Names: []*ast.Ident{ident},
							Type:  typast}
						c.InsertBefore(&ast.DeclStmt{
							Decl: &ast.GenDecl{
								Tok:   token.VAR,
								Specs: []ast.Spec{spec}}})
					}

					// Just patch the node to get the
					// assignment.
					n.Tok = token.ASSIGN

				case token.ADD_ASSIGN, token.SUB_ASSIGN,
					token.MUL_ASSIGN, token.QUO_ASSIGN:
					// Rewrite as lhs = lhs OP rhs (if lhs is
					// computed with side effects you shoot
					// yourself in the foot).
					tok := map[token.Token]token.Token{
						token.ADD_ASSIGN: token.ADD,
						token.SUB_ASSIGN: token.SUB,
						token.MUL_ASSIGN: token.MUL,
						token.QUO_ASSIGN: token.QUO,
					}[n.Tok]
					n.Tok = token.ASSIGN
					expr := &ast.BinaryExpr{
						X:     n.Lhs[0],
						OpPos: n.Pos(),
						Op:    tok,
						Y:     n.Rhs[0],
					}
					n.Rhs[0] = expr
					// We introduced a new expression after
					// typechecking. Let's add it to the
					// type map.
					m.info.Types[expr] = m.info.Types[n.Lhs[0]]
				}
			case *ast.IncDecStmt:
				// Rewrite as expr = expr OP 1
				one := &ast.BasicLit{
					ValuePos: n.Pos(),
					Kind:     token.INT,
					Value:    "1",
				}
				tok := map[token.Token]token.Token{
					token.INC: token.ADD,
					token.DEC: token.SUB,
				}[n.Tok]
				expr := &ast.BinaryExpr{
					X:     n.X,
					OpPos: n.Pos(),
					Op:    tok,
					Y:     one,
				}
				asgn := &ast.AssignStmt{
					Lhs:    []ast.Expr{n.X},
					TokPos: n.Pos(),
					Tok:    token.ASSIGN,
					Rhs:    []ast.Expr{expr},
				}
				// We introduced new expressions after
				// typechecking. Let's add them to the
				// type map.
				m.info.Types[one] = m.info.Types[n.X]
				m.info.Types[expr] = m.info.Types[n.X]
				c.Replace(asgn)
			case *ast.UnaryExpr:
				if n.Op == token.ADD {
					c.Replace(n.X)
				}
			}
			return true
		},
		nil)

	return err
}

// rewrite rewrites the method using tape-writing calls.
func (m *model) rewrite(method *ast.FuncDecl) (err error) {
	// Apply panics on errors. When Apply panics, we return the
	// error as do other functions.
	defer errOnPanic(
        "rewrite",
        &err,
        m.fset.Position(method.Pos()),
    )()

    // ontape switches rewriting on and off. If pre returns true
    // but ontape is false, Apply traverses the children but
    // they are not rewritten (until ontape is true).
	ontape := false
	astutil.Apply(method,
		// pre focuses on the parts of the tree that
		// are to be rewritten.
		func(c *astutil.Cursor) bool {
			n := c.Node()
            if n != nil && n.Pos() != token.NoPos {
                defer errOnPanic(
                    "rewrite/pre",
                    &err,
                    m.fset.Position(n.Pos()),
                )()
            }
			switch n := n.(type) {
			case *ast.BasicLit, *ast.Ident,
				*ast.IndexExpr, *ast.SelectorExpr,
                *ast.StarExpr, *ast.UnaryExpr, *ast.BinaryExpr:
				t, basic := m.info.TypeOf(n.(ast.Expr)).
                    (*types.Basic)
				if !basic || t.Kind() != types.Float64 {
					return false
				}
                if _, ok := n.(*ast.Ident); ok {
                    // SelectorExpr is peculiar: Sel is a child and
                    // implements Expr, but not an expression. I
                    // believe astutil should not traverse Sel at
                    // all.
                    switch c.Parent().(type) {
                    case *ast.SelectorExpr:
                        if strings.Compare(c.Name(), "Sel")==0 {
                            return false
                        }
                    }
                }
            case *ast.CallExpr:
                switch {
                case m.isDifferentiated(n):
                case m.isElemental(n):
                default:
                    // A function which is neither
                    // differentiated nor elemental is called
                    // with all their arguments unmodified.
                    value := callExpr("ad.Value", n)
                    c.Replace(value)
                    return false
                }
			case *ast.ReturnStmt: // if float64
				if len(n.Results) != 1 {
					return false
				}
				for _, r := range n.Results {
					t, basic := m.info.TypeOf(r).(*types.Basic)
					if !basic || t.Kind() != types.Float64 {
						return false
					}
				}
				ontape = true
			case *ast.AssignStmt: // if all are float64
				for _, r := range n.Lhs {
					t, basic := m.info.TypeOf(r).(*types.Basic)
					if !basic || t.Kind() != types.Float64 {
						return false
					}
				}
				ontape = true
			case *ast.ExprStmt: // if a model method call
				call, ok := n.X.(*ast.CallExpr)
				if !ok {
					return false
				}
                if !m.isDifferentiated(call) {
					return false
				}
				ontape = true
			}
			return true
		},

		// post differentiates expressions in bottom-up order.
		func(c *astutil.Cursor) bool {
			if !ontape {
				return true
			}
			n := c.Node()
            if n != nil && n.Pos() != token.NoPos {
                defer errOnPanic(
                    "rewrite/post",
                    &err,
                    m.fset.Position(n.Pos()),
                )()
            }
			switch n := n.(type) {
			case *ast.BasicLit:
				value := callExpr("ad.Value", n)
				c.Replace(value)
			case *ast.Ident, *ast.IndexExpr:
				place := &ast.UnaryExpr{
					Op: token.AND,
					X:  n.(ast.Expr),
				}
				c.Replace(place)
            case *ast.SelectorExpr:
				place := &ast.UnaryExpr{
					Op: token.AND,
					X:  n,
				}
				c.Replace(place)
			case *ast.ReturnStmt:
				ret := callExpr("ad.Return", n.Results...)
				n.Results = []ast.Expr{ret}
				ontape = false
			case *ast.StarExpr:
				c.Replace(n.X)
			case *ast.UnaryExpr:
				switch n.Op {
				case token.SUB:
					neg := callExpr("ad.Arithmetic",
						&ast.BasicLit{
							Kind:  token.INT,
							Value: "ad.OpNeg",
						},
						n.X)
					c.Replace(neg)
				default:
					panic(fmt.Sprintf(
						"cannot rewrite unary %v", n.Op))
				}
			case *ast.BinaryExpr:
				bin := callExpr("ad.Arithmetic",
					&ast.BasicLit{
						Kind: token.INT,
						Value: map[token.Token]string{
							token.ADD: "ad.OpAdd",
							token.SUB: "ad.OpSub",
							token.MUL: "ad.OpMul",
							token.QUO: "ad.OpDiv",
						}[n.Op],
					},
					n.X, n.Y)
				c.Replace(bin)
			case *ast.AssignStmt:
				var asgn ast.Expr
				if len(n.Lhs) == 1 {
					asgn = callExpr("ad.Assignment",
						n.Lhs[0], n.Rhs[0])
				} else {
					asgn = callExpr("ad.ParallelAssignment",
						append(n.Lhs, n.Rhs...)...)
				}
				stmt := &ast.ExprStmt{asgn}
				c.Replace(stmt)
				ontape = false
			case *ast.CallExpr:
                switch {
                case m.isDifferentiated(n):
                case m.isElemental(n):
                    elemental := callExpr("ad.Elemental",
                        append([]ast.Expr{n.Fun}, n.Args...)...)
                    c.Replace(elemental)
                }
			case *ast.ExprStmt:
				ontape = false
			}
			return true
		})

	// Method entry
	// Processed after the traversal so that Apply
	// does not see the added function calls.

	// If we are differentiating Observe, the entry
	// is different than for other methods.
	if strings.Compare(method.Name.Name,
		"Observe") == 0 {
		param := method.Type.Params.List[0]
		var arg ast.Expr
		if strings.Compare(param.Names[0].Name,
			"_") != 0 {
			arg = param.Names[0]
		} else {
            // The parameter is as _; the argument is
            // an empty slice.
			arg = &ast.CompositeLit{
				Type: param.Type,
			}
		}
		setup := &ast.ExprStmt{
			callExpr("ad.Setup", arg),
		}
		method.Body.List = append([]ast.Stmt{setup},
			method.Body.List...)
	} else {
        // Collect float64 parameters. Their values
        // are copied from the tape.
        t := m.info.TypeOf(method.Name).(*types.Signature)
        var params []ast.Expr
        n := t.Params().Len()
        if t.Variadic() {
            // If the function is variadic, 
            // the last parameter is not a float64.
            n-- 
        }
        // Signature parameters are flat, but ast parameters
        // are two-dimensional: a parameter is a Field with
        // possibly multiple names in it.
        iparam, ifield := 0, 0   // ast indices
        for i := 0; i != n; i++ {
            p := t.Params().At(i)
            pt, ok := p.Type().(*types.Basic)
            if !ok || pt.Kind() != types.Float64 {
                continue
            }
            var expr ast.Expr
            if strings.Compare(p.Name(), "_") == 0 {
                // There is no variable to copy the value
                // to, create a dummy value.
                expr = callExpr("ad.Value", &ast.BasicLit{
                    Kind:     token.FLOAT,
                    Value:    "0.",
                })
            } else {
                expr = &ast.UnaryExpr {
                    Op: token.AND,
                    X:  method.Type.Params.List[iparam].
                            Names[ifield],
                }
            }
            params = append(params, expr)
            ifield++
            if ifield == len(method.Type.Params.List[iparam].Names) {
                iparam++
                ifield = 0
            }
        }
        enter := &ast.ExprStmt{
            callExpr("ad.Enter", params...),
        }
        method.Body.List = append([]ast.Stmt{enter},
            method.Body.List...)
	}

	return err
}

// callExpr returns an Expr for call 'name(args...)'.
func callExpr(name string, args ...ast.Expr) ast.Expr {
	return &ast.CallExpr{
		Fun: &ast.Ident{
			Name: name,
		},
		Args: args,
	}
}

// isDifferentiated returns true iff the call
// is of a differentiated method
func (m *model) isDifferentiated(call *ast.CallExpr) bool {
    sel, ok := call.Fun.(*ast.SelectorExpr)
    if !ok {
        return ok
    }
    t, ok := m.info.Selections[sel]
    if !ok {
        return ok
    }
    ok = t.Kind() == types.MethodVal
    if !ok {
        return ok
    }
    ok = t.Recv() != nil && m.isType(t.Recv())
    return ok
}

// isElemental returns true iff the call is of
// an elemental function. It does not check whether
// this is a differentiated function instead and should
// be called after isDifferentiated.
func (m *model) isElemental(call *ast.CallExpr) bool {
    t := m.info.TypeOf(call.Fun).(*types.Signature)
    if t.Results().Len() != 1 {
        return false
    }
	rt, ok := t.Results().At(0).Type().(*types.Basic)
	if !ok {
		return ok
	}
	ok = rt.Kind() == types.Float64 // the result is float64
    if !ok {
        return ok
    }

    if t.Params().Len()==0 {
        return false
    }
    for i := 0; i != t.Params().Len(); i++ {
        pt, ok := t.Params().At(i).Type().(*types.Basic)
        if !ok {
            return ok
        }
        ok = pt.Kind() == types.Float64
        if !ok {
            return ok
        }
    }

    return true
}

// Writing

// write writes the differentiated model as a Go package source.
func (m *model) write() (err error) {
	admpath := path.Join(m.path, "ad")
	// Create the directory for the differentiated model.
	err = os.Mkdir(admpath, os.ModePerm)
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
		f, err := os.Create(path.Join(admpath, fname))
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

// Automatic differentiation of a model
package ad

// A model is defined in it's own package. The model
// must implement interface model.Model. Interface model.DiffModel
// is induced through automatic differentiation.
// In the model's source code:
//   1. Method Observe of interface model.Model is
//      differentiated.
//   2. All methods on the type implementing model.Model
//	are differentiated.
//   3. Within the methods, the following is differentiated:
//	a) assignments to float64;
//      b) returns of float64;
//      c) standalone calls to methods on the object on which
//         Observe was called (apparently called for side
//         effects).
// Functions are considered elementals (and must have a
// registered derivative):
//   a) If they are defined in a different package, and
//   b) Their signature is  func (float64, ...float64) float64;
//      that is, at least one float64 argument and float64 return
//      value.
// Derivatives do not propagate through a function that is not
// an elemental or a call to a method on the model.
// If a derivative is not registered for an elemental, calling
// the elemental in a differentiated context will cause a
// run-time error.

import ()

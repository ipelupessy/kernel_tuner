# Roadmap for the Kernel Tuner

This roadmap presents an overview of the features we are currently planning to
implement. Please note that this is a living document that will evolve as
priorities grow and shift.

### version 0.2.0

This is the list of features that we want to have implemented by the next version.

 * Option to set function that computes search space restriction, instead of a list of strings
 * Option to set function that computes grid dimensions instead of grid divisor lists
 * Option to set dynamically allocated shared memory for CUDA backend
 
### version 1.0.0

These functions are to be implemented by version 1.0.0, but may already be
implemented in earlier versions.

 * Functionality for storing tuning results to datastore on disk
 * Functionality for including auto-tuned kernels in applications
 * Tuning kernels in parallel on a set of nodes in a GPU cluster

### Low priority

These are the things that we would like to implement, but we currently have no
demand for it. If you are interested in any of these, let us know!

 * Provide API for analysis of tuning results
 * Tuning compiler options in combination with other parameters kernel
 * Example that tunes a kernel using thread block re-indexing
 * Example CUDA host code that uses runtime compilation



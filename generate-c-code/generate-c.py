from casadi import (
    MX,
    Function,
    CodeGenerator,
    sin,
    cos
)

x = MX.sym('x',2)

f = Function('f',[x],[sin(x)])
g = Function('g',[x],[cos(x)])

opts = {
    'verbose': True,                # Include comments in generated code
    'main': True,                   # Generate a command line interface
    'mex': False,                   # Generate an MATLAB/Octave MEX entry point
    'cpp': True,                    # Generated code is C++ instead of C
    'casadi_real': 'double',        # Floating point type
    'casadi_int': 'long long int',  # Integer type
    'with_header': False,           # Generate a header file
    'with_mem':	False,              # Generate a simplified C API
    'indent': 2,	                # Number of spaces per indentation level

}
C = CodeGenerator('gen.cpp', opts)
C.add(f)
C.add(g)

C.generate()
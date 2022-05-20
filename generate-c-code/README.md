# Generate C-Code

## Build instructions

Assuming, `gcc` or `g++` is installed, the following command will build the executable.

**Note**: On Windows, use `WSL`.

`gcc --std=c++11 gen.cpp -o gen.o -lm`

* `--std=c++11` Specifies to build for C++11 standard
* `-lm` Specifies to link libraries

## Usage

When running this program each entry of the `[2,1]` vector is entered sequentially.

```bash
./gen.o f
2
3
0.909297 0.14112
```

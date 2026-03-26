# Exceptions

Custom exception classes for clear error handling in LZGraphs.

## Exception Hierarchy

LZGraphs follows a simple exception hierarchy. Custom exceptions are only used where specific recovery logic is likely needed. For general issues (invalid arguments, etc.), standard Python exceptions like `ValueError` and `TypeError` are used with descriptive messages.

```
LZGraphError (base)
├── NoGeneDataError      — gene operation on graph without gene data
├── ConvergenceError     — numerical method did not converge
└── CorruptFileError     — LZG file is corrupt or invalid
```

## Import

```python
from LZGraphs import LZGraphError, NoGeneDataError, ConvergenceError, CorruptFileError
```

## Exception Reference

### LZGraphError
**Base class** for all exceptions raised by the LZGraphs library. Catch this to handle any library-specific error.

```python
try:
    graph.save("path/to/save")
except LZGraphError as e:
    print(f"LZGraphs operation failed: {e}")
```

### NoGeneDataError
Raised when gene-dependent operations (like `sample_genes=True` in `simulate()`) are called on a graph that was built without V/J gene annotations.

```python
try:
    results = graph.simulate(100, sample_genes=True)
except NoGeneDataError:
    print("This graph has no gene data. Falling back to basic simulation.")
    results = graph.simulate(100)
```

### ConvergenceError
Raised by analytical methods like `predicted_richness()` if the underlying numerical solver fails to converge. This can happen at extreme sequencing depths or with highly unusual graph topologies.

### CorruptFileError
Raised by `LZGraph.load()` if the file is not a valid `.lzg` binary, has a checksum mismatch, or was created by an incompatible version of the library.

```python
try:
    graph = LZGraph.load("data.lzg")
except CorruptFileError:
    print("The file is corrupt or not a valid LZGraph.")
```

## Standard Exceptions Used

LZGraphs also uses standard Python exceptions:

- **`ValueError`**: Passed an invalid value (e.g., negative simulation count, unknown variant name).
- **`TypeError`**: Passed an object of the wrong type (e.g., a string instead of a list of sequences).
- **`MemoryError`**: The C backend failed to allocate enough memory for a massive graph or simulation.

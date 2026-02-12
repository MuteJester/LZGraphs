# Exceptions

Custom exception classes for clear error handling in LZGraphs.

## Exception Hierarchy

```
LZGraphError (base)
├── InputValidationError
│   ├── EmptyDataError
│   ├── MissingColumnError
│   └── InvalidSequenceError
├── GraphConstructionError
│   └── EncodingError
├── GeneDataError
│   ├── NoGeneDataError
│   └── GeneAnnotationError
├── WalkError
│   ├── NoValidPathError
│   ├── MissingNodeError
│   └── MissingEdgeError
├── SerializationError
│   ├── UnsupportedFormatError
│   └── CorruptedFileError
├── BOWError
│   ├── EncodingFunctionMismatchError
│   └── UnfittedBOWError
├── GraphOperationError
│   └── IncompatibleGraphsError
└── MetricsError
    └── InsufficientDataError
```

## Import

```python
from LZGraphs.exceptions import (
    LZGraphError,
    MissingColumnError,
    NoGeneDataError,
    MissingNodeError,
    # ... other exceptions
)
```

## Common Exceptions

### MissingColumnError

Raised when a required column is missing.

```python
from LZGraphs.exceptions import MissingColumnError

try:
    graph = AAPLZGraph(data)  # data missing 'cdr3_amino_acid'
except MissingColumnError as e:
    print(f"Missing column: {e.column_name}")
    print(f"Available: {e.available_columns}")
```

### NoGeneDataError

Raised when gene operations are used without gene data.

```python
from LZGraphs.exceptions import NoGeneDataError

try:
    walk, v, j = graph.genomic_random_walk()
except NoGeneDataError as e:
    print(f"Gene data required: {e}")
    # Fall back to regular random walk
    walk = graph.random_walk()
```

### MissingNodeError

Raised when a required node doesn't exist.

```python
from LZGraphs.exceptions import MissingNodeError

try:
    pgen = graph.walk_probability(encoded_sequence)
except MissingNodeError as e:
    print(f"Node not found: {e.node}")
    pgen = 0
```

### EmptyDataError

Raised when input data is empty.

```python
from LZGraphs.exceptions import EmptyDataError

try:
    graph = AAPLZGraph(pd.DataFrame())
except EmptyDataError:
    print("Cannot build graph from empty data")
```

## Error Handling Patterns

### Catch All LZGraphs Errors

```python
from LZGraphs.exceptions import LZGraphError

try:
    # Any LZGraphs operation
    graph = AAPLZGraph(data)
    pgen = graph.walk_probability(encoded)
except LZGraphError as e:
    print(f"LZGraphs error: {e}")
```

### Specific Error Handling

```python
from LZGraphs.exceptions import (
    MissingColumnError,
    NoGeneDataError,
    MissingNodeError
)

try:
    graph = AAPLZGraph(data)
    walk, v, j = graph.genomic_random_walk()
except MissingColumnError as e:
    print(f"Data format error: {e}")
except NoGeneDataError:
    print("Using non-genomic random walk")
    walk = graph.random_walk()
except MissingNodeError as e:
    print(f"Unknown pattern: {e.node}")
```

### Batch Processing with Error Handling

```python
results = []
for seq in sequences:
    try:
        pgen = graph.walk_probability(seq)
        results.append({'sequence': seq, 'pgen': pgen, 'error': None})
    except MissingNodeError as e:
        results.append({'sequence': seq, 'pgen': 0, 'error': str(e)})
    except Exception as e:
        results.append({'sequence': seq, 'pgen': None, 'error': str(e)})

df = pd.DataFrame(results)
print(f"Successful: {df['error'].isna().sum()}")
print(f"Failed: {df['error'].notna().sum()}")
```

## Full Reference

::: LZGraphs.exceptions
    options:
      show_root_heading: false
      show_source: false
      members:
        - LZGraphError
        - InputValidationError
        - EmptyDataError
        - MissingColumnError
        - InvalidSequenceError
        - NoGeneDataError
        - MissingNodeError
        - MissingEdgeError

## See Also

- [Getting Started: Troubleshooting](../getting-started/installation.md#troubleshooting)
- [Resources: FAQ](../resources/faq.md)

"""
Custom exceptions for the LZGraphs library.

This module provides a hierarchy of exception classes that give users
clear, actionable error messages when something goes wrong. Using specific
exception types allows for targeted error handling in downstream code.

Exception Hierarchy:
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
    │   └── MissingNodeError
    ├── SerializationError
    │   └── UnsupportedFormatError
    ├── BOWError
    │   └── EncodingFunctionMismatchError
    └── GraphOperationError
        └── IncompatibleGraphsError

Example:
    >>> from LZGraphs.exceptions import NoGeneDataError, InvalidSequenceError
    >>> try:
    ...     graph.genomic_random_walk()
    ... except NoGeneDataError as e:
    ...     print(f"Gene data required: {e}")
"""


# =============================================================================
# Base Exception
# =============================================================================

class LZGraphError(Exception):
    """
    Base exception for all LZGraphs errors.

    All custom exceptions in this library inherit from this class,
    allowing users to catch all LZGraphs-related errors with a single
    except clause if desired.

    Example:
        >>> try:
        ...     # Any LZGraphs operation
        ...     graph = AAPLZGraph(data)
        ... except LZGraphError as e:
        ...     print(f"LZGraphs error: {e}")
    """
    pass


# =============================================================================
# Input Validation Errors
# =============================================================================

class InputValidationError(LZGraphError):
    """
    Raised when input data fails validation checks.

    This is the base class for all input-related errors. Use more specific
    subclasses when the error type is known.
    """
    pass


class EmptyDataError(InputValidationError):
    """
    Raised when an operation receives empty data where non-empty is required.

    Common causes:
    - Passing an empty DataFrame to graph constructor
    - Passing an empty list of sequences to transform
    - Empty sequence list for diversity metrics

    Example:
        >>> graph = AAPLZGraph(pd.DataFrame())  # Raises EmptyDataError
    """
    pass


class MissingColumnError(InputValidationError):
    """
    Raised when a required column is missing from input DataFrame.

    The error message includes:
    - The name of the missing column
    - The columns that were found in the DataFrame

    Example:
        >>> df = pd.DataFrame({'wrong_col': ['CASS']})
        >>> graph = AAPLZGraph(df)  # Raises MissingColumnError
    """

    def __init__(self, column_name: str, available_columns: list = None, message: str = None):
        if message is None:
            message = f"Required column '{column_name}' not found in DataFrame"
            if available_columns:
                message += f". Available columns: {available_columns}"
        self.column_name = column_name
        self.available_columns = available_columns
        super().__init__(message)


class InvalidSequenceError(InputValidationError):
    """
    Raised when a sequence contains invalid characters or format.

    The error message includes:
    - The problematic sequence (or portion of it)
    - The invalid characters found
    - Expected format information

    Example:
        >>> graph.walk_probability("INVALID123SEQUENCE")  # Raises InvalidSequenceError
    """

    def __init__(self, sequence: str = None, invalid_chars: str = None, message: str = None):
        if message is None:
            message = "Invalid sequence"
            if sequence:
                display_seq = sequence[:50] + "..." if len(sequence) > 50 else sequence
                message += f": '{display_seq}'"
            if invalid_chars:
                message += f". Invalid characters: {invalid_chars}"
        self.sequence = sequence
        self.invalid_chars = invalid_chars
        super().__init__(message)


class InvalidProbabilityError(InputValidationError):
    """
    Raised when probability values are invalid.

    Common causes:
    - Probabilities don't sum to 1.0
    - Negative probability values
    - Probability array length mismatch

    Example:
        >>> choice(['A', 'B'], [0.3, 0.3])  # Raises InvalidProbabilityError (sum != 1)
    """

    def __init__(self, message: str = None, prob_sum: float = None):
        if message is None and prob_sum is not None:
            message = f"Probabilities must sum to ~1.0, got {prob_sum:.4f}"
        elif message is None:
            message = "Invalid probability distribution"
        self.prob_sum = prob_sum
        super().__init__(message)


# =============================================================================
# Graph Construction Errors
# =============================================================================

class GraphConstructionError(LZGraphError):
    """
    Raised when graph construction fails.

    This is the base class for errors that occur during the process
    of building an LZGraph from sequence data.
    """
    pass


class EncodingError(GraphConstructionError):
    """
    Raised when sequence encoding into subpatterns fails.

    This typically occurs when:
    - The sequence contains unsupported characters
    - The encoding function encounters an unexpected pattern
    - Position calculation fails

    Example:
        >>> graph.encode_sequence("???")  # May raise EncodingError
    """

    def __init__(self, sequence: str = None, message: str = None):
        if message is None:
            message = "Failed to encode sequence"
            if sequence:
                display_seq = sequence[:30] + "..." if len(sequence) > 30 else sequence
                message += f": '{display_seq}'"
        self.sequence = sequence
        super().__init__(message)


# =============================================================================
# Gene Data Errors
# =============================================================================

class GeneDataError(LZGraphError):
    """
    Base class for gene-related errors.

    These errors occur when working with V/J gene annotations
    in genetic LZGraphs.
    """
    pass


class NoGeneDataError(GeneDataError):
    """
    Raised when a gene-related operation is attempted on a non-genetic graph.

    This occurs when:
    - Calling genomic_random_walk() on a graph with genetic=False
    - Accessing gene prediction features without gene data
    - Attempting gene-based filtering without annotations

    Solution:
        Build the graph with V and J gene columns in the input DataFrame.

    Example:
        >>> graph = AAPLZGraph(df_without_genes)
        >>> graph.genomic_random_walk()  # Raises NoGeneDataError
    """

    def __init__(self, operation: str = None, message: str = None):
        if message is None:
            message = "This operation requires gene annotation data (genetic=True)"
            if operation:
                message = f"'{operation}' requires gene annotation data (genetic=True)"
            message += ". Build the graph with V and J gene columns to enable this feature."
        self.operation = operation
        super().__init__(message)


class GeneAnnotationError(GeneDataError):
    """
    Raised when gene annotation data is malformed or inconsistent.

    This can occur when:
    - V/J gene names don't match expected patterns
    - Gene data is missing from edge attributes
    - Inconsistent gene annotations across edges

    Example:
        >>> graph.walk_genes(walk)  # May raise GeneAnnotationError if data corrupt
    """
    pass


# =============================================================================
# Walk and Probability Errors
# =============================================================================

class WalkError(LZGraphError):
    """
    Base class for errors during graph traversal operations.

    These errors occur during random walks, probability calculations,
    or path-finding operations on the graph.
    """
    pass


class NoValidPathError(WalkError):
    """
    Raised when no valid path exists for a given operation.

    This can occur when:
    - A random walk cannot proceed (no outgoing edges)
    - No path exists between start and end nodes
    - All potential paths are blocked

    Example:
        >>> graph.random_walk()  # May raise NoValidPathError if graph disconnected
    """

    def __init__(self, start_node: str = None, message: str = None):
        if message is None:
            message = "No valid path found"
            if start_node:
                message += f" from node '{start_node}'"
        self.start_node = start_node
        super().__init__(message)


class MissingNodeError(WalkError):
    """
    Raised when a required node does not exist in the graph.

    This typically occurs when:
    - Computing walk probability for an unseen sequence
    - A subpattern in the sequence was never observed during training
    - Referencing a node that was removed

    Example:
        >>> graph.walk_probability("CASSXYZABC")  # Raises MissingNodeError if XYZ never seen
    """

    def __init__(self, node: str = None, message: str = None):
        if message is None:
            message = "Node not found in graph"
            if node:
                message = f"Node '{node}' not found in graph"
        self.node = node
        super().__init__(message)


class MissingEdgeError(WalkError):
    """
    Raised when a required edge does not exist in the graph.

    This occurs when:
    - A transition between two nodes was never observed
    - Computing probability for an impossible transition

    Example:
        >>> # If 'CA_0' -> 'XY_1' was never seen during training
        >>> graph.walk_probability("CAXY...")  # May raise MissingEdgeError
    """

    def __init__(self, source: str = None, target: str = None, message: str = None):
        if message is None:
            if source and target:
                message = f"Edge '{source}' -> '{target}' not found in graph"
            else:
                message = "Edge not found in graph"
        self.source = source
        self.target = target
        super().__init__(message)


# =============================================================================
# Serialization Errors
# =============================================================================

class SerializationError(LZGraphError):
    """
    Base class for errors during save/load operations.

    These errors occur when saving graphs to files or loading
    them back into memory.
    """
    pass


class UnsupportedFormatError(SerializationError):
    """
    Raised when an unsupported serialization format is specified.

    Supported formats are:
    - 'pickle': Binary format (recommended)
    - 'json': Human-readable format

    Example:
        >>> graph.save('file.xml', format='xml')  # Raises UnsupportedFormatError
    """

    def __init__(self, format: str = None, supported: list = None, message: str = None):
        if message is None:
            message = f"Unsupported format: '{format}'"
            if supported:
                message += f". Supported formats: {supported}"
            else:
                message += ". Supported formats: ['pickle', 'json']"
        self.format = format
        self.supported = supported or ['pickle', 'json']
        super().__init__(message)


class CorruptedFileError(SerializationError):
    """
    Raised when a saved graph file appears to be corrupted.

    This can occur when:
    - File was partially written
    - File was modified externally
    - Incompatible version loaded

    Example:
        >>> graph = AAPLZGraph.load('corrupted.pkl')  # Raises CorruptedFileError
    """
    pass


# =============================================================================
# BOW (Bag of Words) Errors
# =============================================================================

class BOWError(LZGraphError):
    """
    Base class for Bag of Words encoder errors.

    These errors occur during BOW fitting, transformation,
    or combination operations.
    """
    pass


class EncodingFunctionMismatchError(BOWError):
    """
    Raised when combining BOW objects with different encoding functions.

    BOW objects can only be combined (using +) if they use the same
    encoding function. This ensures the resulting dictionary is consistent.

    Example:
        >>> bow1 = LZBOW(encoding_function=lempel_ziv_decomposition)
        >>> bow2 = LZBOW(encoding_function=lambda x: list(x))
        >>> combined = bow1 + bow2  # Raises EncodingFunctionMismatchError
    """
    pass


class UnfittedBOWError(BOWError):
    """
    Raised when transform is called on an unfitted BOW object.

    The BOW encoder must be fitted with fit() before calling transform().

    Example:
        >>> bow = LZBOW()
        >>> bow.transform("CASSABC")  # Raises UnfittedBOWError if not fitted
    """
    pass


# =============================================================================
# Graph Operation Errors
# =============================================================================

class GraphOperationError(LZGraphError):
    """
    Base class for errors during graph operations.

    These errors occur during operations like graph union,
    comparison, or modification.
    """
    pass


class IncompatibleGraphsError(GraphOperationError):
    """
    Raised when attempting to combine incompatible graphs.

    Graphs must be of the same type to be combined. For example,
    you cannot union an AAPLZGraph with an NDPLZGraph.

    Example:
        >>> graph_union(aap_graph, ndp_graph)  # Raises IncompatibleGraphsError
    """

    def __init__(self, type1: str = None, type2: str = None, message: str = None):
        if message is None:
            if type1 and type2:
                message = f"Cannot combine graphs of different types: {type1} and {type2}"
            else:
                message = "Cannot combine graphs of different types"
        self.type1 = type1
        self.type2 = type2
        super().__init__(message)


# =============================================================================
# Metrics Errors
# =============================================================================

class MetricsError(LZGraphError):
    """
    Base class for errors in metrics calculations.

    These errors occur during diversity, entropy, or other
    statistical metric computations.
    """
    pass


class InsufficientDataError(MetricsError):
    """
    Raised when there's not enough data for a statistical calculation.

    This can occur when:
    - K-diversity requires more sequences than available
    - Confidence intervals need more samples
    - Statistical tests need more data points

    Example:
        >>> k_diversity(sequences, k=1000, draws=100)  # Raises if < 1000 sequences
    """

    def __init__(self, required: int = None, available: int = None, message: str = None):
        if message is None:
            message = "Insufficient data for calculation"
            if required and available:
                message = f"Insufficient data: need at least {required}, got {available}"
        self.required = required
        self.available = available
        super().__init__(message)


# =============================================================================
# Convenience exports
# =============================================================================

__all__ = [
    # Base
    'LZGraphError',

    # Input validation
    'InputValidationError',
    'EmptyDataError',
    'MissingColumnError',
    'InvalidSequenceError',
    'InvalidProbabilityError',

    # Graph construction
    'GraphConstructionError',
    'EncodingError',

    # Gene data
    'GeneDataError',
    'NoGeneDataError',
    'GeneAnnotationError',

    # Walk/probability
    'WalkError',
    'NoValidPathError',
    'MissingNodeError',
    'MissingEdgeError',

    # Serialization
    'SerializationError',
    'UnsupportedFormatError',
    'CorruptedFileError',

    # BOW
    'BOWError',
    'EncodingFunctionMismatchError',
    'UnfittedBOWError',

    # Graph operations
    'GraphOperationError',
    'IncompatibleGraphsError',

    # Metrics
    'MetricsError',
    'InsufficientDataError',
]

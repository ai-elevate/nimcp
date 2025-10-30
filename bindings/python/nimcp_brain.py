"""
NIMCP Brain API - Python Bindings

High-level Python interface to NIMCP's lightweight neural learning framework.

Example usage:
    from nimcp_brain import Brain, BrainSize, BrainTask

    # Create a small brain for classification
    brain = Brain("ethics_decision", BrainSize.SMALL, BrainTask.CLASSIFICATION,
                  num_inputs=4, num_outputs=3)

    # Learn from examples
    features = [0.8, 0.3, 0.5, 0.6]
    brain.learn_example(features, "block", confidence=0.9)

    # Make decisions
    decision = brain.decide(features)
    print(f"Decision: {decision.label} (confidence: {decision.confidence:.2f})")

    # Save trained brain
    brain.save("ethics_brain.nimcp")
"""

import ctypes
import os
import platform
from enum import IntEnum
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass

# Find the NIMCP library
def _find_library():
    """Locate the NIMCP shared library"""
    lib_name = {
        'Linux': 'libnimcp_core.so',
        'Darwin': 'libnimcp_core.dylib',
        'Windows': 'nimcp_core.dll'
    }.get(platform.system(), 'libnimcp_core.so')

    # Search paths
    search_paths = [
        os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'src', 'lib'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'lib'),
        '/usr/local/lib',
        '/usr/lib'
    ]

    for path in search_paths:
        lib_path = os.path.join(path, lib_name)
        if os.path.exists(lib_path):
            return lib_path

    # Try system search
    return lib_name

# Load library
_lib = ctypes.CDLL(_find_library())

#=============================================================================
# Enumerations
#=============================================================================

class BrainSize(IntEnum):
    """Brain size presets"""
    TINY = 0    # 100 neurons, <1MB
    SMALL = 1   # 1K neurons, ~10MB
    MEDIUM = 2  # 10K neurons, ~50MB
    LARGE = 3   # 100K neurons, ~500MB
    CUSTOM = 4  # User-defined

class BrainTask(IntEnum):
    """Brain task templates"""
    CLASSIFICATION = 0
    REGRESSION = 1
    PATTERN_MATCHING = 2
    SEQUENCE = 3
    ASSOCIATION = 4
    CUSTOM = 5

#=============================================================================
# Structures
#=============================================================================

class _BrainDecision(ctypes.Structure):
    """Internal C structure for brain_decision_t"""
    _fields_ = [
        ('label', ctypes.c_char * 64),
        ('confidence', ctypes.c_float),
        ('output_vector', ctypes.POINTER(ctypes.c_float)),
        ('output_size', ctypes.c_uint32),
        ('num_active_neurons', ctypes.c_uint32),
        ('active_neuron_ids', ctypes.POINTER(ctypes.c_uint32)),
        ('sparsity', ctypes.c_float),
        ('explanation', ctypes.c_char * 256),
        ('inference_time_us', ctypes.c_uint64),
    ]

class _BrainStats(ctypes.Structure):
    """Internal C structure for brain_stats_t"""
    _fields_ = [
        ('task_name', ctypes.c_char * 64),
        ('size', ctypes.c_int),
        ('num_neurons', ctypes.c_uint32),
        ('num_synapses', ctypes.c_uint32),
        ('num_active_synapses', ctypes.c_uint32),
        ('total_inferences', ctypes.c_uint64),
        ('total_learning_steps', ctypes.c_uint64),
        ('avg_sparsity', ctypes.c_float),
        ('avg_inference_time_us', ctypes.c_float),
        ('current_learning_rate', ctypes.c_float),
        ('accuracy', ctypes.c_float),
        ('memory_bytes', ctypes.c_size_t),
    ]

#=============================================================================
# Python-friendly classes
#=============================================================================

@dataclass
class Decision:
    """Decision result from brain inference"""
    label: str
    confidence: float
    output_vector: List[float]
    num_active_neurons: int
    active_neuron_ids: List[int]
    sparsity: float
    explanation: str
    inference_time_us: int

    @property
    def inference_time_ms(self) -> float:
        """Get inference time in milliseconds"""
        return self.inference_time_us / 1000.0

@dataclass
class BrainStats:
    """Brain statistics"""
    task_name: str
    size: BrainSize
    num_neurons: int
    num_synapses: int
    num_active_synapses: int
    total_inferences: int
    total_learning_steps: int
    avg_sparsity: float
    avg_inference_time_us: float
    current_learning_rate: float
    accuracy: float
    memory_bytes: int

    @property
    def avg_inference_time_ms(self) -> float:
        """Get average inference time in milliseconds"""
        return self.avg_inference_time_us / 1000.0

    @property
    def memory_mb(self) -> float:
        """Get memory usage in megabytes"""
        return self.memory_bytes / (1024 * 1024)

#=============================================================================
# C Function Signatures
#=============================================================================

# brain_create
_lib.brain_create.argtypes = [
    ctypes.c_char_p,  # task_name
    ctypes.c_int,     # size
    ctypes.c_int,     # task
    ctypes.c_uint32,  # num_inputs
    ctypes.c_uint32,  # num_outputs
]
_lib.brain_create.restype = ctypes.c_void_p

# brain_destroy
_lib.brain_destroy.argtypes = [ctypes.c_void_p]
_lib.brain_destroy.restype = None

# brain_learn_example
_lib.brain_learn_example.argtypes = [
    ctypes.c_void_p,                # brain
    ctypes.POINTER(ctypes.c_float), # features
    ctypes.c_uint32,                # num_features
    ctypes.c_char_p,                # label
    ctypes.c_float,                 # confidence
]
_lib.brain_learn_example.restype = ctypes.c_float

# brain_decide
_lib.brain_decide.argtypes = [
    ctypes.c_void_p,                # brain
    ctypes.POINTER(ctypes.c_float), # features
    ctypes.c_uint32,                # num_features
]
_lib.brain_decide.restype = ctypes.POINTER(_BrainDecision)

# brain_free_decision
_lib.brain_free_decision.argtypes = [ctypes.POINTER(_BrainDecision)]
_lib.brain_free_decision.restype = None

# brain_save
_lib.brain_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.brain_save.restype = ctypes.c_bool

# brain_load
_lib.brain_load.argtypes = [ctypes.c_char_p]
_lib.brain_load.restype = ctypes.c_void_p

# brain_get_stats
_lib.brain_get_stats.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(_BrainStats)
]
_lib.brain_get_stats.restype = ctypes.c_bool

# brain_get_top_neurons
_lib.brain_get_top_neurons.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(ctypes.c_float),
]
_lib.brain_get_top_neurons.restype = ctypes.c_uint32

# brain_prune
_lib.brain_prune.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.brain_prune.restype = ctypes.c_uint32

# brain_get_last_error
_lib.brain_get_last_error.argtypes = []
_lib.brain_get_last_error.restype = ctypes.c_char_p

#=============================================================================
# Python Brain Class
#=============================================================================

class Brain:
    """
    High-level Python interface to NIMCP Brain API

    A Brain is a lightweight neural network that can learn patterns from
    examples or from external teachers (like LLMs) and make fast local decisions.

    Args:
        task_name: Human-readable name for the brain
        size: Brain size preset (BrainSize enum)
        task: Task type (BrainTask enum)
        num_inputs: Number of input features
        num_outputs: Number of output classes/values

    Example:
        brain = Brain("my_classifier", BrainSize.SMALL, BrainTask.CLASSIFICATION,
                      num_inputs=10, num_outputs=3)
    """

    def __init__(self, task_name: str, size: BrainSize, task: BrainTask,
                 num_inputs: int, num_outputs: int):
        self._handle = _lib.brain_create(
            task_name.encode('utf-8'),
            int(size),
            int(task),
            num_inputs,
            num_outputs
        )

        if not self._handle:
            error = _lib.brain_get_last_error()
            error_msg = error.decode('utf-8') if error else "Unknown error"
            raise RuntimeError(f"Failed to create brain: {error_msg}")

        self.task_name = task_name
        self.size = size
        self.task = task
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    def __del__(self):
        """Clean up brain resources"""
        if hasattr(self, '_handle') and self._handle:
            _lib.brain_destroy(self._handle)

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.__del__()

    def learn_example(self, features: List[float], label: str,
                     confidence: float = 1.0) -> float:
        """
        Learn from a single labeled example

        Args:
            features: Input feature vector
            label: Target label/class
            confidence: Training weight (0.0-1.0)

        Returns:
            Loss value for this example

        Example:
            loss = brain.learn_example([0.5, 0.8, 0.3], "class_A", confidence=0.9)
        """
        if len(features) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} features, got {len(features)}")

        features_array = (ctypes.c_float * len(features))(*features)

        loss = _lib.brain_learn_example(
            self._handle,
            features_array,
            len(features),
            label.encode('utf-8'),
            confidence
        )

        if loss < 0:
            error = _lib.brain_get_last_error()
            error_msg = error.decode('utf-8') if error else "Unknown error"
            raise RuntimeError(f"Learning failed: {error_msg}")

        return loss

    def learn_batch(self, examples: List[Tuple[List[float], str, float]]) -> float:
        """
        Learn from a batch of examples

        Args:
            examples: List of (features, label, confidence) tuples

        Returns:
            Average loss over batch

        Example:
            examples = [
                ([0.1, 0.2], "A", 1.0),
                ([0.8, 0.9], "B", 0.9),
            ]
            avg_loss = brain.learn_batch(examples)
        """
        total_loss = 0.0
        for features, label, confidence in examples:
            loss = self.learn_example(features, label, confidence)
            total_loss += loss

        return total_loss / len(examples)

    def decide(self, features: List[float]) -> Decision:
        """
        Make a decision for input features

        Args:
            features: Input feature vector

        Returns:
            Decision object with label, confidence, and interpretability info

        Example:
            decision = brain.decide([0.5, 0.8, 0.3])
            print(f"{decision.label}: {decision.confidence:.2f}")
        """
        if len(features) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} features, got {len(features)}")

        features_array = (ctypes.c_float * len(features))(*features)

        c_decision = _lib.brain_decide(self._handle, features_array, len(features))

        if not c_decision:
            error = _lib.brain_get_last_error()
            error_msg = error.decode('utf-8') if error else "Unknown error"
            raise RuntimeError(f"Decision failed: {error_msg}")

        # Convert to Python-friendly Decision object
        decision = Decision(
            label=c_decision.contents.label.decode('utf-8'),
            confidence=c_decision.contents.confidence,
            output_vector=[c_decision.contents.output_vector[i]
                          for i in range(c_decision.contents.output_size)],
            num_active_neurons=c_decision.contents.num_active_neurons,
            active_neuron_ids=[c_decision.contents.active_neuron_ids[i]
                              for i in range(c_decision.contents.num_active_neurons)],
            sparsity=c_decision.contents.sparsity,
            explanation=c_decision.contents.explanation.decode('utf-8'),
            inference_time_us=c_decision.contents.inference_time_us,
        )

        # Free C memory
        _lib.brain_free_decision(c_decision)

        return decision

    def save(self, filepath: str) -> bool:
        """
        Save trained brain to file

        Args:
            filepath: Path to save file

        Returns:
            True on success

        Example:
            brain.save("my_brain.nimcp")
        """
        success = _lib.brain_save(self._handle, filepath.encode('utf-8'))

        if not success:
            error = _lib.brain_get_last_error()
            error_msg = error.decode('utf-8') if error else "Unknown error"
            raise RuntimeError(f"Save failed: {error_msg}")

        return True

    @staticmethod
    def load(filepath: str) -> 'Brain':
        """
        Load trained brain from file

        Args:
            filepath: Path to load from

        Returns:
            Loaded Brain instance

        Example:
            brain = Brain.load("my_brain.nimcp")
        """
        handle = _lib.brain_load(filepath.encode('utf-8'))

        if not handle:
            error = _lib.brain_get_last_error()
            error_msg = error.decode('utf-8') if error else "Unknown error"
            raise RuntimeError(f"Load failed: {error_msg}")

        # Create Brain instance without calling __init__
        brain = Brain.__new__(Brain)
        brain._handle = handle

        # Get stats to populate attributes
        stats = brain.get_stats()
        brain.task_name = stats.task_name
        brain.size = stats.size
        # Note: num_inputs/outputs not stored, would need to add to metadata

        return brain

    def get_stats(self) -> BrainStats:
        """
        Get brain statistics

        Returns:
            BrainStats object with performance metrics

        Example:
            stats = brain.get_stats()
            print(f"Trained on {stats.total_learning_steps} examples")
            print(f"Inference time: {stats.avg_inference_time_ms:.2f} ms")
        """
        c_stats = _BrainStats()

        success = _lib.brain_get_stats(self._handle, ctypes.byref(c_stats))

        if not success:
            error = _lib.brain_get_last_error()
            error_msg = error.decode('utf-8') if error else "Unknown error"
            raise RuntimeError(f"Get stats failed: {error_msg}")

        return BrainStats(
            task_name=c_stats.task_name.decode('utf-8'),
            size=BrainSize(c_stats.size),
            num_neurons=c_stats.num_neurons,
            num_synapses=c_stats.num_synapses,
            num_active_synapses=c_stats.num_active_synapses,
            total_inferences=c_stats.total_inferences,
            total_learning_steps=c_stats.total_learning_steps,
            avg_sparsity=c_stats.avg_sparsity,
            avg_inference_time_us=c_stats.avg_inference_time_us,
            current_learning_rate=c_stats.current_learning_rate,
            accuracy=c_stats.accuracy,
            memory_bytes=c_stats.memory_bytes,
        )

    def get_top_neurons(self, top_n: int = 10) -> List[Tuple[int, float]]:
        """
        Get most important neurons

        Args:
            top_n: Number of top neurons to return

        Returns:
            List of (neuron_id, importance) tuples

        Example:
            top_neurons = brain.get_top_neurons(5)
            for neuron_id, importance in top_neurons:
                print(f"Neuron {neuron_id}: importance={importance:.4f}")
        """
        neuron_ids = (ctypes.c_uint32 * top_n)()
        importances = (ctypes.c_float * top_n)()

        count = _lib.brain_get_top_neurons(
            self._handle,
            top_n,
            neuron_ids,
            importances
        )

        return [(neuron_ids[i], importances[i]) for i in range(count)]

    def prune(self, threshold: float = 0.01) -> int:
        """
        Prune weak connections

        Args:
            threshold: Prune synapses with weight < threshold

        Returns:
            Number of synapses pruned

        Example:
            pruned = brain.prune(0.05)
            print(f"Pruned {pruned} weak connections")
        """
        return _lib.brain_prune(self._handle, threshold)

    def __repr__(self):
        stats = self.get_stats()
        return (f"Brain(name='{self.task_name}', size={self.size.name}, "
                f"neurons={stats.num_neurons}, "
                f"inferences={stats.total_inferences})")

#=============================================================================
# Convenience Functions
#=============================================================================

def create_classifier(name: str, num_inputs: int, num_outputs: int,
                     size: BrainSize = BrainSize.SMALL) -> Brain:
    """
    Create a classification brain with sensible defaults

    Example:
        brain = create_classifier("spam_detector", num_inputs=100, num_outputs=2)
    """
    return Brain(name, size, BrainTask.CLASSIFICATION, num_inputs, num_outputs)

def create_pattern_matcher(name: str, num_inputs: int,
                          size: BrainSize = BrainSize.SMALL) -> Brain:
    """
    Create a pattern matching brain

    Example:
        brain = create_pattern_matcher("anomaly_detector", num_inputs=50)
    """
    return Brain(name, size, BrainTask.PATTERN_MATCHING, num_inputs, 1)

__all__ = [
    'Brain',
    'BrainSize',
    'BrainTask',
    'Decision',
    'BrainStats',
    'create_classifier',
    'create_pattern_matcher',
]

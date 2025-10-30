/**
 * @file nimcp-brain.ts
 * @brief TypeScript/JavaScript bindings for NIMCP Brain API
 *
 * Uses node-ffi-napi for FFI bindings to the C library.
 *
 * Installation:
 *   npm install ffi-napi ref-napi ref-array-napi ref-struct-napi
 *
 * Example usage:
 * ```typescript
 * import { Brain, BrainSize, BrainTask } from './nimcp-brain';
 *
 * const brain = new Brain("ethics", BrainSize.SMALL, BrainTask.CLASSIFICATION, 4, 3);
 *
 * brain.learnExample([0.8, 0.3, 0.5, 0.6], "block", 0.9);
 *
 * const decision = brain.decide([0.8, 0.3, 0.5, 0.6]);
 * console.log(`Decision: ${decision.label} (confidence: ${decision.confidence})`);
 *
 * brain.save("brain.nimcp");
 * ```
 */

import ffi from 'ffi-napi';
import ref from 'ref-napi';
import StructType from 'ref-struct-napi';
import ArrayType from 'ref-array-napi';
import * as path from 'path';
import * as os from 'os';

//=============================================================================
// Native Types
//=============================================================================

const float = ref.types.float;
const uint32 = ref.types.uint32;
const uint64 = ref.types.uint64;
const size_t = ref.types.size_t;
const bool = ref.types.bool;
const charPtr = ref.refType(ref.types.char);
const voidPtr = ref.refType(ref.types.void);
const floatPtr = ref.refType(float);
const uint32Ptr = ref.refType(uint32);

const FloatArray = ArrayType(float);
const Uint32Array = ArrayType(uint32);

//=============================================================================
// Enumerations
//=============================================================================

export enum BrainSize {
    TINY = 0,
    SMALL = 1,
    MEDIUM = 2,
    LARGE = 3,
    CUSTOM = 4
}

export enum BrainTask {
    CLASSIFICATION = 0,
    REGRESSION = 1,
    PATTERN_MATCHING = 2,
    SEQUENCE = 3,
    ASSOCIATION = 4,
    CUSTOM = 5
}

//=============================================================================
// Structures
//=============================================================================

const BrainDecisionStruct = StructType({
    label: ArrayType(ref.types.char, 64),
    confidence: float,
    output_vector: floatPtr,
    output_size: uint32,
    num_active_neurons: uint32,
    active_neuron_ids: uint32Ptr,
    sparsity: float,
    explanation: ArrayType(ref.types.char, 256),
    inference_time_us: uint64,
});

const BrainStatsStruct = StructType({
    task_name: ArrayType(ref.types.char, 64),
    size: ref.types.int,
    num_neurons: uint32,
    num_synapses: uint32,
    num_active_synapses: uint32,
    total_inferences: uint64,
    total_learning_steps: uint64,
    avg_sparsity: float,
    avg_inference_time_us: float,
    current_learning_rate: float,
    accuracy: float,
    memory_bytes: size_t,
});

//=============================================================================
// Load Library
//=============================================================================

function findLibrary(): string {
    const libName = {
        'linux': 'libnimcp_core.so',
        'darwin': 'libnimcp_core.dylib',
        'win32': 'nimcp_core.dll'
    }[os.platform()] || 'libnimcp_core.so';

    const searchPaths = [
        path.join(__dirname, '..', '..', 'build', 'src', 'lib', libName),
        path.join(__dirname, '..', '..', 'lib', libName),
        path.join('/usr/local/lib', libName),
        path.join('/usr/lib', libName),
    ];

    for (const libPath of searchPaths) {
        try {
            require('fs').accessSync(libPath);
            return libPath;
        } catch {}
    }

    return libName; // Try system search
}

const lib = ffi.Library(findLibrary(), {
    'brain_create': [voidPtr, [charPtr, 'int', 'int', uint32, uint32]],
    'brain_destroy': ['void', [voidPtr]],
    'brain_learn_example': [float, [voidPtr, floatPtr, uint32, charPtr, float]],
    'brain_decide': [ref.refType(BrainDecisionStruct), [voidPtr, floatPtr, uint32]],
    'brain_free_decision': ['void', [ref.refType(BrainDecisionStruct)]],
    'brain_save': [bool, [voidPtr, charPtr]],
    'brain_load': [voidPtr, [charPtr]],
    'brain_get_stats': [bool, [voidPtr, ref.refType(BrainStatsStruct)]],
    'brain_get_top_neurons': [uint32, [voidPtr, uint32, uint32Ptr, floatPtr]],
    'brain_prune': [uint32, [voidPtr, float]],
    'brain_get_last_error': [charPtr, []],
});

//=============================================================================
// TypeScript Interfaces
//=============================================================================

export interface Decision {
    label: string;
    confidence: number;
    outputVector: number[];
    numActiveNeurons: number;
    activeNeuronIds: number[];
    sparsity: number;
    explanation: string;
    inferenceTimeUs: number;
    inferenceTimeMs: number;
}

export interface Stats {
    taskName: string;
    size: BrainSize;
    numNeurons: number;
    numSynapses: number;
    numActiveSynapses: number;
    totalInferences: number;
    totalLearningSteps: number;
    avgSparsity: number;
    avgInferenceTimeUs: number;
    avgInferenceTimeMs: number;
    currentLearningRate: number;
    accuracy: number;
    memoryBytes: number;
    memoryMb: number;
}

export interface NeuronImportance {
    neuronId: number;
    importance: number;
}

//=============================================================================
// Brain Class
//=============================================================================

/**
 * High-level TypeScript interface to NIMCP Brain API
 *
 * A Brain is a lightweight neural network that can learn patterns from
 * examples or from external teachers (like LLMs) and make fast local decisions.
 */
export class Brain {
    private handle: any;
    public readonly taskName: string;
    public readonly size: BrainSize;
    public readonly task: BrainTask;
    public readonly numInputs: number;
    public readonly numOutputs: number;

    /**
     * Create a new brain
     *
     * @param taskName Human-readable name for the brain
     * @param size Brain size preset
     * @param task Task type
     * @param numInputs Number of input features
     * @param numOutputs Number of output classes/values
     */
    constructor(
        taskName: string,
        size: BrainSize,
        task: BrainTask,
        numInputs: number,
        numOutputs: number
    ) {
        const nameBuffer = Buffer.from(taskName + '\0', 'utf-8');

        this.handle = lib.brain_create(nameBuffer, size, task, numInputs, numOutputs);

        if (this.handle.isNull()) {
            const errorPtr = lib.brain_get_last_error();
            const error = errorPtr.isNull() ? 'Unknown error' : ref.readCString(errorPtr, 0);
            throw new Error(`Failed to create brain: ${error}`);
        }

        this.taskName = taskName;
        this.size = size;
        this.task = task;
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
    }

    /**
     * Load brain from file
     *
     * @param filepath Path to brain file
     * @returns Loaded Brain instance
     */
    static load(filepath: string): Brain {
        const pathBuffer = Buffer.from(filepath + '\0', 'utf-8');
        const handle = lib.brain_load(pathBuffer);

        if (handle.isNull()) {
            const errorPtr = lib.brain_get_last_error();
            const error = errorPtr.isNull() ? 'Unknown error' : ref.readCString(errorPtr, 0);
            throw new Error(`Failed to load brain: ${error}`);
        }

        // Create Brain instance without calling constructor
        const brain = Object.create(Brain.prototype);
        brain.handle = handle;

        // Get stats to populate properties
        const stats = brain.getStats();
        brain.taskName = stats.taskName;
        brain.size = stats.size;

        return brain;
    }

    /**
     * Clean up brain resources
     */
    destroy(): void {
        if (this.handle && !this.handle.isNull()) {
            lib.brain_destroy(this.handle);
            this.handle = null;
        }
    }

    /**
     * Learn from a single labeled example
     *
     * @param features Input feature vector
     * @param label Target label/class
     * @param confidence Training weight (0.0-1.0)
     * @returns Loss value for this example
     */
    learnExample(features: number[], label: string, confidence: number = 1.0): number {
        if (features.length !== this.numInputs) {
            throw new Error(`Expected ${this.numInputs} features, got ${features.length}`);
        }

        const featuresArray = new FloatArray(features);
        const labelBuffer = Buffer.from(label + '\0', 'utf-8');

        const loss = lib.brain_learn_example(
            this.handle,
            featuresArray.buffer,
            features.length,
            labelBuffer,
            confidence
        );

        if (loss < 0) {
            const errorPtr = lib.brain_get_last_error();
            const error = errorPtr.isNull() ? 'Unknown error' : ref.readCString(errorPtr, 0);
            throw new Error(`Learning failed: ${error}`);
        }

        return loss;
    }

    /**
     * Learn from a batch of examples
     *
     * @param examples Array of [features, label, confidence] tuples
     * @returns Average loss over batch
     */
    learnBatch(examples: Array<[number[], string, number]>): number {
        let totalLoss = 0;

        for (const [features, label, confidence] of examples) {
            totalLoss += this.learnExample(features, label, confidence);
        }

        return totalLoss / examples.length;
    }

    /**
     * Make a decision for input features
     *
     * @param features Input feature vector
     * @returns Decision object with label, confidence, and interpretability info
     */
    decide(features: number[]): Decision {
        if (features.length !== this.numInputs) {
            throw new Error(`Expected ${this.numInputs} features, got ${features.length}`);
        }

        const featuresArray = new FloatArray(features);

        const decisionPtr = lib.brain_decide(this.handle, featuresArray.buffer, features.length);

        if (decisionPtr.isNull()) {
            const errorPtr = lib.brain_get_last_error();
            const error = errorPtr.isNull() ? 'Unknown error' : ref.readCString(errorPtr, 0);
            throw new Error(`Decision failed: ${error}`);
        }

        const cDecision = decisionPtr.deref();

        // Convert C struct to TypeScript object
        const label = ref.readCString(cDecision.label, 0);
        const explanation = ref.readCString(cDecision.explanation, 0);

        const outputVector: number[] = [];
        for (let i = 0; i < cDecision.output_size; i++) {
            outputVector.push(cDecision.output_vector.readFloatLE(i * 4));
        }

        const activeNeuronIds: number[] = [];
        for (let i = 0; i < cDecision.num_active_neurons; i++) {
            activeNeuronIds.push(cDecision.active_neuron_ids.readUInt32LE(i * 4));
        }

        const decision: Decision = {
            label,
            confidence: cDecision.confidence,
            outputVector,
            numActiveNeurons: cDecision.num_active_neurons,
            activeNeuronIds,
            sparsity: cDecision.sparsity,
            explanation,
            inferenceTimeUs: Number(cDecision.inference_time_us),
            inferenceTimeMs: Number(cDecision.inference_time_us) / 1000.0,
        };

        // Free C memory
        lib.brain_free_decision(decisionPtr);

        return decision;
    }

    /**
     * Save trained brain to file
     *
     * @param filepath Path to save file
     */
    save(filepath: string): void {
        const pathBuffer = Buffer.from(filepath + '\0', 'utf-8');

        const success = lib.brain_save(this.handle, pathBuffer);

        if (!success) {
            const errorPtr = lib.brain_get_last_error();
            const error = errorPtr.isNull() ? 'Unknown error' : ref.readCString(errorPtr, 0);
            throw new Error(`Save failed: ${error}`);
        }
    }

    /**
     * Get brain statistics
     *
     * @returns Stats object with performance metrics
     */
    getStats(): Stats {
        const cStats = new BrainStatsStruct();

        const success = lib.brain_get_stats(this.handle, cStats.ref());

        if (!success) {
            const errorPtr = lib.brain_get_last_error();
            const error = errorPtr.isNull() ? 'Unknown error' : ref.readCString(errorPtr, 0);
            throw new Error(`Get stats failed: ${error}`);
        }

        return {
            taskName: ref.readCString(cStats.task_name, 0),
            size: cStats.size as BrainSize,
            numNeurons: cStats.num_neurons,
            numSynapses: cStats.num_synapses,
            numActiveSynapses: cStats.num_active_synapses,
            totalInferences: Number(cStats.total_inferences),
            totalLearningSteps: Number(cStats.total_learning_steps),
            avgSparsity: cStats.avg_sparsity,
            avgInferenceTimeUs: cStats.avg_inference_time_us,
            avgInferenceTimeMs: cStats.avg_inference_time_us / 1000.0,
            currentLearningRate: cStats.current_learning_rate,
            accuracy: cStats.accuracy,
            memoryBytes: Number(cStats.memory_bytes),
            memoryMb: Number(cStats.memory_bytes) / (1024 * 1024),
        };
    }

    /**
     * Get most important neurons
     *
     * @param topN Number of top neurons to return
     * @returns Array of neuron importance rankings
     */
    getTopNeurons(topN: number = 10): NeuronImportance[] {
        const neuronIds = new Uint32Array(topN);
        const importances = new FloatArray(topN);

        const count = lib.brain_get_top_neurons(
            this.handle,
            topN,
            neuronIds.buffer,
            importances.buffer
        );

        const result: NeuronImportance[] = [];
        for (let i = 0; i < count; i++) {
            result.push({
                neuronId: neuronIds[i],
                importance: importances[i],
            });
        }

        return result;
    }

    /**
     * Prune weak connections
     *
     * @param threshold Prune synapses with weight < threshold
     * @returns Number of synapses pruned
     */
    prune(threshold: number = 0.01): number {
        return lib.brain_prune(this.handle, threshold);
    }
}

//=============================================================================
// Convenience Functions
//=============================================================================

/**
 * Create a classification brain with sensible defaults
 */
export function createClassifier(
    name: string,
    numInputs: number,
    numOutputs: number,
    size: BrainSize = BrainSize.SMALL
): Brain {
    return new Brain(name, size, BrainTask.CLASSIFICATION, numInputs, numOutputs);
}

/**
 * Create a pattern matching brain
 */
export function createPatternMatcher(
    name: string,
    numInputs: number,
    size: BrainSize = BrainSize.SMALL
): Brain {
    return new Brain(name, size, BrainTask.PATTERN_MATCHING, numInputs, 1);
}

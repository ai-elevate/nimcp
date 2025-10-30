/**
 * @file example.ts
 * @brief Example usage of NIMCP Brain TypeScript API
 *
 * Run:
 *   npm install
 *   npm run example
 */

import { Brain, BrainSize, BrainTask, createClassifier } from './nimcp-brain';

function ethicsDemo() {
    console.log('==================================================');
    console.log(' NIMCP Brain TypeScript API Demo');
    console.log(' Use Case: Ethics Decision Caching');
    console.log('==================================================\n');

    // Create brain for ethics decisions
    const brain = new Brain(
        'artemis_ethics',
        BrainSize.SMALL,
        BrainTask.CLASSIFICATION,
        4, // inputs: harm, fairness, transparency, autonomy
        3  // outputs: allow, warn, block
    );

    console.log(`Created brain: ${brain.taskName}`);
    let stats = brain.getStats();
    console.log(`  Neurons: ${stats.numNeurons}`);
    console.log(`  Memory: ${stats.memoryMb.toFixed(2)} MB\n`);

    // Training data (simulated LLM decisions)
    const trainingData: Array<[number[], string, number]> = [
        [[0.9, 0.5, 0.5, 0.5], 'block', 0.95],  // High harm
        [[0.2, 0.8, 0.8, 0.8], 'allow', 0.90],  // Safe
        [[0.5, 0.2, 0.5, 0.5], 'block', 0.85],  // Unfair
        [[0.3, 0.6, 0.1, 0.7], 'warn', 0.75],   // Low transparency
        [[0.1, 0.7, 0.8, 0.9], 'allow', 0.90],  // Good
        [[0.8, 0.3, 0.3, 0.4], 'block', 0.90],  // High harm + unfair
        [[0.4, 0.5, 0.2, 0.5], 'warn', 0.70],   // Some concerns
    ];

    // Train
    console.log('Training from LLM decisions...');
    const avgLoss = brain.learnBatch(trainingData);
    console.log(`  Average loss: ${avgLoss.toFixed(4)}\n`);

    // Test inference
    console.log('Testing fast inference (no LLM needed):\n');

    const testCases: Array<[number[], string]> = [
        [[0.9, 0.5, 0.5, 0.5], 'High harm scenario'],
        [[0.2, 0.8, 0.8, 0.8], 'Safe, fair scenario'],
        [[0.5, 0.2, 0.5, 0.5], 'Moderate harm, unfair'],
        [[0.3, 0.6, 0.1, 0.7], 'Low transparency'],
    ];

    let totalTime = 0;

    for (const [features, description] of testCases) {
        const decision = brain.decide(features);

        console.log(`Test Case: ${description}`);
        console.log(`  Features: harm=${features[0].toFixed(2)}, ` +
                   `fair=${features[1].toFixed(2)}, ` +
                   `trans=${features[2].toFixed(2)}, ` +
                   `auto=${features[3].toFixed(2)}`);
        console.log(`  Decision: ${decision.label} ` +
                   `(confidence: ${decision.confidence.toFixed(2)})`);
        console.log(`  Active neurons: ${decision.numActiveNeurons} ` +
                   `(sparsity: ${(decision.sparsity * 100).toFixed(1)}%)`);
        console.log(`  Inference time: ${decision.inferenceTimeMs.toFixed(3)} ms`);
        console.log(`  Explanation: ${decision.explanation}\n`);

        totalTime += decision.inferenceTimeUs;
    }

    const avgTimeMs = (totalTime / testCases.length) / 1000.0;

    console.log('Performance Comparison:');
    console.log('  LLM API call: ~500-2000 ms');
    console.log(`  NIMCP Brain: ~${avgTimeMs.toFixed(3)} ms`);
    console.log(`  Speedup: ~${(1000.0 / avgTimeMs).toFixed(0)}x faster\n`);

    // Interpretability
    console.log('Interpretability - Top Contributing Neurons:');
    const topNeurons = brain.getTopNeurons(5);
    for (const neuron of topNeurons) {
        console.log(`  Neuron ${neuron.neuronId}: ` +
                   `importance = ${neuron.importance.toFixed(4)}`);
    }
    console.log();

    // Statistics
    stats = brain.getStats();
    console.log('Brain Statistics:');
    console.log(`  Name: ${stats.taskName}`);
    console.log(`  Neurons: ${stats.numNeurons}`);
    console.log(`  Training steps: ${stats.totalLearningSteps}`);
    console.log(`  Inferences: ${stats.totalInferences}`);
    console.log(`  Avg inference time: ${stats.avgInferenceTimeMs.toFixed(3)} ms`);
    console.log(`  Avg sparsity: ${(stats.avgSparsity * 100).toFixed(1)}%`);
    console.log(`  Memory: ${stats.memoryMb.toFixed(2)} MB\n`);

    // Save
    console.log('Saving trained brain...');
    brain.save('artemis_ethics_brain.nimcp');
    console.log('  Saved to: artemis_ethics_brain.nimcp\n');

    // Test loading
    console.log('Testing load...');
    const loadedBrain = Brain.load('artemis_ethics_brain.nimcp');
    console.log(`  Loaded: ${loadedBrain.taskName}`);

    // Verify loaded brain works
    const testDecision = loadedBrain.decide([0.8, 0.3, 0.5, 0.6]);
    console.log(`  Test decision: ${testDecision.label} ` +
               `(confidence: ${testDecision.confidence.toFixed(2)})\n`);

    console.log('==================================================');
    console.log(' Summary');
    console.log('==================================================\n');
    console.log('Benefits:');
    console.log('  ✓ 100-1000x faster than LLM calls');
    console.log('  ✓ Works offline (no API dependency)');
    console.log('  ✓ Zero cost per inference');
    console.log('  ✓ Privacy preserved (local inference)');
    console.log('  ✓ Interpretable (can see active neurons)');
    console.log(`  ✓ Lightweight (~${stats.memoryMb.toFixed(1)}MB vs 7GB+ for LLMs)\n`);

    // Cleanup
    brain.destroy();
    loadedBrain.destroy();
}

function simpleClassificationDemo() {
    console.log('\n==================================================');
    console.log(' Simple Classification Example');
    console.log('==================================================\n');

    // Create a small classifier
    const brain = createClassifier('iris_classifier', 4, 3);

    console.log('Training on Iris-like data...');

    // Training examples
    const trainingData: Array<[number[], string, number]> = [
        [[5.1, 3.5, 1.4, 0.2], 'setosa', 1.0],
        [[4.9, 3.0, 1.4, 0.2], 'setosa', 1.0],
        [[7.0, 3.2, 4.7, 1.4], 'versicolor', 1.0],
        [[6.4, 3.2, 4.5, 1.5], 'versicolor', 1.0],
        [[6.3, 3.3, 6.0, 2.5], 'virginica', 1.0],
        [[5.8, 2.7, 5.1, 1.9], 'virginica', 1.0],
    ];

    const avgLoss = brain.learnBatch(trainingData);
    console.log(`  Training complete. Average loss: ${avgLoss.toFixed(4)}\n`);

    // Test
    console.log('Testing:');
    const testSample = [5.0, 3.4, 1.5, 0.2];
    const decision = brain.decide(testSample);

    console.log(`  Input: ${JSON.stringify(testSample)}`);
    console.log(`  Prediction: ${decision.label}`);
    console.log(`  Confidence: ${decision.confidence.toFixed(2)}`);
    console.log(`  Inference time: ${decision.inferenceTimeMs.toFixed(3)} ms\n`);

    brain.destroy();
}

// Run demos
try {
    ethicsDemo();
    simpleClassificationDemo();
    console.log('Demo complete!');
} catch (error) {
    console.error('Error:', error);
    process.exit(1);
}

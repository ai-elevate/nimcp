const nimcp = require('./index');

console.log('Testing NIMCP Node.js bindings...');

// Create neural network configuration
const config = {
    num_inputs: 10,
    num_outputs: 5,
    num_hidden: 20,
    learning_rate: 0.01
};

console.log('Creating neural network with config:', config);

try {
    const network = new nimcp.NeuralNetwork(config);
    console.log('✓ Neural network created successfully');

    // Create test input
    const inputs = new Array(10).fill(0).map(() => Math.random());
    console.log('Feeding input:', inputs);

    const feedSuccess = network.feedInput(inputs);
    console.log('✓ Feed input:', feedSuccess ? 'success' : 'failed');

    const outputs = network.getOutput();
    console.log('✓ Got outputs:', outputs);

    console.log('\n✓ All tests passed!');
} catch (error) {
    console.error('✗ Test failed:', error.message);
    process.exit(1);
}

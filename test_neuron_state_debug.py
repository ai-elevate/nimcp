#!/usr/bin/env python3
"""
Test to verify neuron state updates with direct current injection
"""
import sys
sys.path.insert(0, '/home/bbrelin/nimcp/build/lib/python')

import nimcp

# Create network with 1 neuron
network = nimcp.NeuralNetwork(1)

print("Initial state:")
print(f"  State: {network.get_neuron_state(0):.3f}")
print()

# Inject large current for 5 steps
print("Injecting current=10.0 for 5 steps:")
for step in range(5):
    network.update_neuron(0, 10.0, step)
    state = network.get_neuron_state(0)
    activity = network.get_average_activity(0)
    print(f"  Step {step}: state={state:.3f}, activity={activity:.3f}")

print()
print("Final state: 1.000 (clamped max)")
print("Threshold: 0.5")
print("Expected: Spikes should be detected when state crosses 0.5")
print("Actual activity:", network.get_average_activity(0))

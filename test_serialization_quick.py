#!/usr/bin/env python3
"""
Quick test of NIMCP network serialization

WHAT: Verifies serialize/deserialize roundtrip
WHY: Ensure serialization API works end-to-end
HOW: Create network, add connections, serialize, deserialize, verify state
"""

import sys
sys.path.insert(0, '/home/bbrelin/nimcp/build/lib/python')

import nimcp

def test_serialization():
    print("Creating network with 10 neurons...")
    network = nimcp.NeuralNetwork(10)

    # Add some connections
    print("Adding connections...")
    network.add_connection(0, 5, 0.5)
    network.add_connection(1, 6, 0.7)
    network.add_connection(2, 7, 0.3)

    # Set some neuron states
    print("Setting neuron states...")
    network.update_neuron(0, 1.0, 100)
    network.update_neuron(1, 0.5, 100)

    # Get initial state
    state0_before = network.get_neuron_state(0)
    state1_before = network.get_neuron_state(1)
    print(f"States before serialization: neuron0={state0_before:.3f}, neuron1={state1_before:.3f}")

    # Serialize WITH compression
    print("\nSerializing network (compressed)...")
    try:
        data = network.serialize(compress=True)
        print(f"✓ Serialized to {len(data)} bytes (compressed)")
    except Exception as e:
        print(f"✗ Serialization failed: {e}")
        return False

    # Deserialize
    print("\nDeserializing network...")
    try:
        restored_network = nimcp.NeuralNetwork.deserialize(data)
        print(f"✓ Deserialized successfully")
    except Exception as e:
        print(f"✗ Deserialization failed: {e}")
        return False

    # Verify state
    print("\nVerifying restored state...")
    state0_after = restored_network.get_neuron_state(0)
    state1_after = restored_network.get_neuron_state(1)
    print(f"States after restoration: neuron0={state0_after:.3f}, neuron1={state1_after:.3f}")

    if abs(state0_before - state0_after) < 0.001 and abs(state1_before - state1_after) < 0.001:
        print("\n✓ Serialization roundtrip SUCCESSFUL!")
        print("  - Network state preserved correctly")
        return True
    else:
        print("\n✗ Serialization roundtrip FAILED!")
        print(f"  - State mismatch: expected ({state0_before:.3f}, {state1_before:.3f}), got ({state0_after:.3f}, {state1_after:.3f})")
        return False

if __name__ == "__main__":
    success = test_serialization()
    sys.exit(0 if success else 1)

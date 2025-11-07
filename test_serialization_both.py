#!/usr/bin/env python3
"""Test both compressed and uncompressed serialization"""

import sys
sys.path.insert(0, '/home/bbrelin/nimcp/build/lib/python')
import nimcp

def test_mode(compress):
    mode = "compressed" if compress else "uncompressed"
    print(f"\n{'='*60}")
    print(f"Testing {mode} serialization")
    print('='*60)

    # Create network
    network = nimcp.NeuralNetwork(10)
    network.add_connection(0, 5, 0.5)
    network.add_connection(1, 6, 0.7)
    network.update_neuron(0, 1.0, 100)
    network.update_neuron(1, 0.5, 100)

    state0_before = network.get_neuron_state(0)
    state1_before = network.get_neuron_state(1)

    # Serialize
    data = network.serialize(compress=compress)
    print(f"Serialized to {len(data)} bytes")

    # Show first 20 bytes in hex
    hex_data = ' '.join(f'{b:02x}' for b in data[:20])
    print(f"First 20 bytes: {hex_data}")

    # Deserialize
    try:
        restored = nimcp.NeuralNetwork.deserialize(data)
        state0_after = restored.get_neuron_state(0)
        state1_after = restored.get_neuron_state(1)

        if abs(state0_before - state0_after) < 0.001 and abs(state1_before - state1_after) < 0.001:
            print(f"✓ {mode.upper()} serialization SUCCESS")
            return True
        else:
            print(f"✗ {mode.upper()} state mismatch")
            return False
    except Exception as e:
        print(f"✗ {mode.upper()} deserialization failed: {e}")
        return False

if __name__ == "__main__":
    results = []
    results.append(("uncompressed", test_mode(False)))
    results.append(("compressed", test_mode(True)))

    print(f"\n{'='*60}")
    print("SUMMARY:")
    print('='*60)
    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{name:15s}: {status}")

    sys.exit(0 if all(r[1] for r in results) else 1)

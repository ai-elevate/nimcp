"""Minimal repro for learn() crash under GDB."""
import nimcp, sys
nimcp.init()
print("Creating brain with 1.5M neurons...", flush=True)
brain = nimcp.Brain(
    "Athena",
    nimcp.BRAIN_LARGE,
    nimcp.TASK_CLASSIFICATION,
    1024, 256,
    neuron_count=1500000,
)
print(f"Brain created: {brain.get_neuron_count()} neurons", flush=True)
result = brain.predict([0.5]*1024)
print(f"Predict OK", flush=True)
print("Calling brain.learn()...", flush=True)
sys.stdout.flush()
loss = brain.learn([0.5]*1024, "test:label_0")
print(f"Learn OK: loss={loss}", flush=True)

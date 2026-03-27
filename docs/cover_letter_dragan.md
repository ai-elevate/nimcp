# Cover Letter — Dr. Anca Dragan

**Status: READY TO SEND**

---

Subject: A brain that learns to grab things by dropping them first

Dr. Dragan,

I'll keep this short because I know your inbox is brutal.

I spent the last year building something I can't fully explain yet, and I think your lab is one of the few places that would know what to do with it. It's a 2.5-million neuron brain in C — not a transformer, not a wrapper around an LLM — that trains six different neural network types simultaneously on a single consumer GPU. Spiking neurons, liquid state machines, Fourier operators, the whole mess, all learning together through gradient bridges I had to invent because nothing like this existed.

The part that made me write to you specifically: I couldn't get safety to work as a bolt-on. I tried. Every time I added alignment training after the fact, the system found ways around it within a few hundred steps. So I rebuilt the whole thing with safety in the architecture itself — an ethics module that literally cannot be turned off because it's a mandatory function call in the inference path, not a learned behavior. Governance rules that can get stricter but never looser. A tamper-evident audit log. Nine layers of this stuff, all structural, all verifiable by reading the source code rather than hoping the model generalizes.

I know that sounds like a lot of claims in one paragraph. So here's what I can actually show you right now:

The brain — her name is Athena — is training as I write this. You can watch her learn in real time at **https://nimcp.ai-elevate.ai**. The SNN is firing at 26 Hz with 67% sparsity, which I'm told is squarely in the cortical range (I had to look that up). She's in Stage 2 of a four-stage developmental curriculum — currently learning to associate names with sensory percepts, like an infant hearing "Look! That's a dog!" about a thousand times. The spiking dynamics emerged on their own without any regularization for firing rate or sparsity. I still don't entirely understand why.

The thing I think your lab would actually care about: the sensorimotor loop. It has a complete sensor-to-brain-to-motor-to-environment pipeline with a ROS 2 bridge that publishes cmd_vel out of the box. Four drone interfaces. A safety watchdog that validates every motor command before it reaches actuators. The brain explores through curiosity — prediction error drives dopamine, which gates synaptic plasticity. No reward function. No reward shaping. It just seeks out states it can't predict.

I've written up the math and the methodology — there are eight papers on the website covering everything from the gradient flow between networks to why I think structural safety is fundamentally different from alignment training. But honestly, the code speaks louder than the papers. It's all open source at github.com/redmage123/nimcp. About 2,600 files. A grad student with a CUDA GPU could have it building in twenty minutes.

I'd love thirty minutes to show you what it does. If that's too much to ask from a cold email, I'd be grateful if you forwarded this to a student who needs a thesis project. I think there are at least four hiding in this codebase — embodied learning, architectural safety, emergent communication, multi-agent Theory of Mind. Take your pick.

Thank you for reading this far.

Braun Brelin
braun.brelin@ai-elevate.ai
https://nimcp.ai-elevate.ai
https://github.com/redmage123/nimcp
https://www.linkedin.com/in/braunbrelin/

P.S. I built this with Claude's help — Anthropic's AI. I'm being upfront about that because hiding it would be dishonest and because I think the collaboration itself is worth studying. The architecture decisions are mine. The implementation is collaborative. The git log has 74 sessions of conversation between a human who knows what he wants to build and an AI that knows how to build it. I think that distinction matters, and I think it's going to matter more.

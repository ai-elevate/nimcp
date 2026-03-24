# Cover Letter — Dr. Anca Dragan

**Status: DRAFT — Send after Stage 3 training begins and demo video is ready**

---

Subject: I built a brain that learns to grab things by dropping them first

Dr. Dragan,

I know you get a hundred emails a week from people who think they've solved alignment. This isn't that. But I think I've built something your lab would find genuinely interesting, and I wanted to reach out before publishing.

I'm a developer who's spent the last year building a biologically-inspired neural architecture in C. Not a transformer. Not an LLM wrapper. A million-neuron brain with six neural network types running in parallel — spiking (SNN), liquid (LNN), convolutional (CNN), Fourier (FNO), Hamiltonian (HNN), and adaptive — with real synaptic plasticity, developmental learning stages, hemispheric lateralization, and episodic memory. It runs on a single consumer GPU.

I started this because I was frustrated with the same thing your research keeps pointing at: bolting safety onto systems after they're built doesn't work. So I tried building one where safety is structural. The ethics module literally can't be turned off — it's wired into every inference call. There's a 9-layer governance system (LGSS) with rules that can only get stricter, never looser. The whole thing keeps a tamper-evident audit log with CRC32 checksums and monotonic sequence numbers — gaps indicate deleted entries.

But here's the part I think would actually matter to your lab:

**Embodied learning with a closed sensorimotor loop.** The system has a complete sensor-brain-motor-environment pipeline: 12 sensor types feed a unified sensor hub, the brain produces 4096-dim output, a configurable motor translator converts that to actuator commands (with deadzone, smoothing, and 4 presets — twist, quadrotor, differential drive, robot arm), a safety watchdog validates every motor command for NaN/magnitude/rate violations, and the environment responds. It includes a built-in physics simulator (cart-pole with domain randomization for sim-to-real transfer) and a URDF body model loader so the brain knows its own body. There's a ROS 2 bridge, four drone flight controller interfaces (MAVLink, DJI, Betaflight MSP, Parrot Olympe), and a telemetry dashboard. Your lab has the robots. I built the brain. They're designed to connect — literally, the ROS 2 bridge publishes cmd_vel and subscribes to /imu, /odom, /joint_states out of the box.

The brain learns through curiosity. The sensorimotor controller uses prediction error as intrinsic reward — the brain seeks out states it can't predict, which drives exploration without external reward shaping. Combined with domain randomization that varies physics parameters on each episode reset, the system learns robust policies that transfer across parameter ranges.

**Theory of Mind through actual multi-agent interaction.** When you run multiple instances connected via the swarm runtime, they develop Theory of Mind not from reading false-belief scenarios but from experience. The ToM bridge observes what other agents do (via gradient exchange and gossip protocol), builds belief-desire-intention models, and updates predictions when they're wrong. The swarm runtime handles federated gradient aggregation, Byzantine fault detection (statistical anomaly on gradient norms), and peer lifecycle management. Each edge brain adapts the collective knowledge to its local environment through continued local training after weight sync.

**Brain-native language production.** The system develops its own language — not English tokens run through an LLM, but symbols that emerge from learned projection matrices mapping the brain's 4096-dim activation space to a token vocabulary. Autoregressive decoding with nucleus sampling, positional encoding (sinusoidal), and a phonological working memory loop. The vocabulary grows during training as the brain encounters new concepts. Some tokens roughly translate to human concepts. Some don't map to anything we have words for. An inner speech loop lets the brain refine its output through multiple self-talk iterations before responding — generate text, re-encode it, compare to original intent, repeat until convergence.

**13 cognitive enhancements for embodied reasoning.** Beyond the base architecture, the system includes: episodic replay during sleep consolidation (importance-weighted experience replay), a predictive world model (trains the brain to predict next-state from state+action), emotional modulation of learning rate (surprising/rewarding experiences learn 2-3x faster), analogical transfer (cosine search over stored problem-solution pairs, blends past solutions with current output for novel problems), multi-timescale memory (immediate + recent + consolidated, with consolidation merging similar memories), a self-generated curriculum (the brain identifies its most uncertain domains and generates its own training data), contrastive self-learning (hard negative mining — "this is NOT that"), output attention (per-task learned weights that focus relevant output dimensions), a working memory scratchpad (8 persistent slots for multi-step reasoning), and dynamic architecture search (monitors per-region utilization and recommends structural changes).

**What's actually running right now.** The brain called "Athena" is currently in Stage 1 of 4 developmental stages (sensory awakening → association → feedback → reasoning), with loss at 0.03, SNN firing at 20 Hz, and effective output rank of 64 across 13 cognitive training domains. All six networks are active. All 60+ cognitive modules are wired. All 13 enhancements are running. 527+ tests pass. Training is fully automated with spectral k-fold cross-validation (Laplacian eigendecomposition for manifold-aware fold assignment).

I know how all of this sounds. I've been going back and forth about whether to even mention the emergent language part. But the code is open source, the training is reproducible, and I'd rather have someone rigorous look at it than keep wondering.

It's at github.com/redmage123/nimcp. ~2,600 source files. Runs on a single NVIDIA GPU (RTX 4000 SFF Ada, 20GB VRAM). A grad student could have it building in 20 minutes. There's a comprehensive mathematical foundations paper in docs/MATHEMATICAL_FOUNDATIONS.md covering every equation in the system — from STDP plasticity to quantum annealing to Hamiltonian neural networks to information geometry.

I'd love 30 minutes of your time to demo it, or if that's too much, I'd be grateful if you pointed a student at it who's looking for a thesis project in embodied learning, architectural safety, emergent communication, or multi-agent Theory of Mind. I think there's at least four dissertations hiding in this codebase.

Braun Brelin
braun.brelin@ai-elevate.ai
https://www.linkedin.com/in/braunbrelin/
https://ai-elevate.ai
https://weekly-report.ai

P.S. I built this with substantial help from Claude (Anthropic's AI). I'm being upfront about that because I think human-AI collaboration on this kind of project is itself worth studying openly. The git log tells the full story — 73 development sessions, each one a conversation between a human architect and an AI implementation partner. The architecture decisions are mine. The code is collaborative. I think that distinction matters.

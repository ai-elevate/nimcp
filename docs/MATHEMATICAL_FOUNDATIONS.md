# Mathematical Foundations of NIMCP

*Neuro-Inspired Modular Control Protocol -- A Formal Treatment*

**Version 2.6.4 | March 2026**

---

## Abstract

This paper presents the complete mathematical framework underlying the Neuro-Inspired Modular Control Protocol (NIMCP), a 2-million-neuron artificial brain system with six network types, 60+ cognitive modules, and full biological plasticity. We formalize the neural dynamics (LIF, LNN, SNN, HNN), learning rules (STDP, BCM, homeostatic, adjoint), spectral methods (FFT, FNO), quantum-inspired algorithms (annealing, walks, Monte Carlo), information-theoretic measures (Shannon, PID, IIT), differential geometry (Riemannian manifolds, Fisher information, surface minimization), positional encoding schemes, and safety mathematics. Every equation corresponds to implemented code in the NIMCP codebase, with source file references provided throughout.

---

## 1. Neural Dynamics

### 1.1 Leaky Integrate-and-Fire Model

The membrane potential of neuron $i$ evolves according to:

$$\tau_m \frac{dV_i}{dt} = -(V_i - V_{\text{rest}}) + R_m I_i(t)$$

where $\tau_m$ is the membrane time constant, $V_{\text{rest}}$ the resting potential, $R_m$ the membrane resistance, and $I_i(t)$ the total synaptic input current. A spike is emitted when $V_i \geq V_{\text{thresh}}$, after which $V_i \leftarrow V_{\text{reset}}$ and the neuron enters a refractory period $t_{\text{ref}}$.

The SNN populations in NIMCP use this model with per-neuron state vectors tracking membrane voltage, refractory timers, and spike indicators. The FNO bridge (Section 3.4) learns to approximate these dynamics at the population level.

*Source: `include/snn/nimcp_snn_types.h`, `src/snn/nimcp_snn_step.c`*

### 1.2 Spiking Neural Network Population Dynamics

Synaptic current into neuron $i$ from population $p$ is:

$$I_i^{\text{syn}}(t) = \sum_{j \in \text{pre}(i)} w_{ji} \sum_{k} \alpha(t - t_j^k)$$

where $w_{ji}$ is the synaptic weight, $t_j^k$ the $k$-th spike time of presynaptic neuron $j$, and $\alpha(t)$ is a synaptic kernel (exponential or alpha function). Spike trains are stored in ring buffers with fields `total_spikes` and per-population spike counts, indexed by `n_populations`.

*Source: `include/snn/nimcp_snn_types.h`, `include/snn/nimcp_snn_training.h`*

### 1.3 Liquid Neural Network (LNN)

The LNN implements continuous-time recurrent dynamics via an ODE:

$$\frac{dx_i}{dt} = -\frac{x_i}{\tau_i} + f\bigl(W_{\text{rec}} \, x + W_{\text{in}} \, I(t) + b\bigr)_i$$

where $x_i$ is the hidden state, $\tau_i$ a learned time constant (clamped to $\tau_{\text{safe}} \geq 0.01$ to prevent $1/\tau^2$ explosion), $W_{\text{rec}}$ the recurrent weight matrix, $W_{\text{in}}$ the input projection, and $f$ a nonlinearity (typically $\tanh$). Gradients are computed via the adjoint method (Section 2.6) with per-layer tensors `grad_W_rec`, `grad_tau_base`, and `grad_b_in` serving as the authoritative gradient storage.

*Source: `include/lnn/nimcp_lnn_types.h`, `src/lnn/nimcp_lnn_forward.c`*

### 1.4 Hamiltonian Neural Networks (HNN)

Energy-conserving dynamics are achieved by learning a Hamiltonian function $H: \mathbb{R}^{2n} \to \mathbb{R}$ and deriving dynamics from Hamilton's equations:

$$\frac{dq}{dt} = \frac{\partial H}{\partial p}, \qquad \frac{dp}{dt} = -\frac{\partial H}{\partial q}$$

NIMCP supports two forms:

**Separable Hamiltonian:**

$$H(q, p) = T(p) + V(q)$$

where $T$ is kinetic energy (parameterized by an MLP over $p$) and $V$ is potential energy (MLP over $q$), enabling cheaper gradient computation.

**General Hamiltonian:**

$$H(q, p) = \text{MLP}([q; p])$$

with a multi-layer network of configurable depth (`n_hidden_layers`, default 2) and width (`hidden_dim`, default $2n$).

Energy conservation $dH/dt = 0$ is enforced by the **Stormer-Verlet symplectic integrator**:

$$\begin{aligned}
p_{1/2} &= p_n - \frac{\Delta t}{2} \frac{\partial H}{\partial q}\bigg|_{q_n, p_n} \\
q_{n+1} &= q_n + \Delta t \, \frac{\partial H}{\partial p}\bigg|_{q_n, p_{1/2}} \\
p_{n+1} &= p_{1/2} - \frac{\Delta t}{2} \frac{\partial H}{\partial q}\bigg|_{q_{n+1}, p_{1/2}}
\end{aligned}$$

External input drives momentum via a coupling parameter: $p \leftarrow p + \gamma \cdot I_{\text{ext}}$, where $\gamma$ is `input_coupling`. Energy deviation is monitored as $|H(t) - H(0)| / |H(0)|$ with a regularization penalty weighted by `energy_penalty_weight`.

The connection to the Free Energy Principle is: $H \equiv F$ (variational free energy).

*Source: `include/lnn/nimcp_lnn_hamiltonian.h`, `src/lnn/nimcp_lnn_hamiltonian.c`*

---

## 2. Learning Rules

### 2.1 Spike-Timing-Dependent Plasticity (STDP)

The weight change follows the classic Bi & Poo (1998) exponential window:

$$\Delta w = \begin{cases} A_+ \exp\!\left(-\frac{\Delta t}{\tau_+}\right) & \text{if } \Delta t > 0 \text{ (pre before post, LTP)} \\ -A_- \exp\!\left(\frac{\Delta t}{\tau_-}\right) & \text{if } \Delta t < 0 \text{ (post before pre, LTD)} \end{cases}$$

Default parameters: $A_+ = 0.005$, $A_- = 0.00525$, $\tau_+ = \tau_- = 20$ ms, $w \in [0, w_{\max}]$ with $w_{\max} = 1.0$.

**Trace-based implementation:** Instead of storing all spike times, exponentially decaying traces are maintained:

$$\begin{aligned}
\text{pre\_trace}(t + \Delta t) &= \text{pre\_trace}(t) \cdot \exp(-\Delta t / \tau_+) \\
\text{post\_trace}(t + \Delta t) &= \text{post\_trace}(t) \cdot \exp(-\Delta t / \tau_-)
\end{aligned}$$

Traces below `NIMCP_DENORMAL_THRESHOLD` ($10^{-10}$) are flushed to zero for performance.

**Three-factor learning with dopamine modulation:** The effective learning rate is:

$$\eta_{\text{eff}} = \eta_{\text{base}} \times (1 + g_{\text{DA}} \cdot [\text{DA}]) \times m_{\text{burst}}$$

where $g_{\text{DA}} = 100.0$ (default), $[\text{DA}]$ is dopamine concentration, and $m_{\text{burst}} = 3.0$ during phasic dopamine bursts.

*Source: `include/plasticity/stdp/nimcp_stdp.h`, `src/plasticity/stdp/nimcp_stdp.c`*

### 2.2 BCM Rule (Bienenstock-Cooper-Munro)

The BCM sliding threshold rule for cortical learning:

$$\begin{aligned}
\Delta w &= \eta \cdot y \cdot (y - \theta) \cdot x \\
\frac{d\theta}{dt} &= \frac{y^2 - \theta}{\tau_\theta}
\end{aligned}$$

where $y$ is postsynaptic activity, $x$ is presynaptic activity, $\theta$ is the sliding modification threshold, and $\tau_\theta$ is the threshold time constant. When $y > \theta$, the synapse potentiates; when $y < \theta$, it depresses. The threshold tracks the mean squared activity $\langle y^2 \rangle$, making the rule self-stabilizing without explicit normalization.

*Source: `include/plasticity/bcm/nimcp_bcm.h`, `src/plasticity/bcm/nimcp_bcm.c`*

### 2.3 Eligibility Traces

Eligibility traces bridge the temporal credit assignment gap between Hebbian coincidence and delayed reward:

$$e_{ij}(t+\Delta t) = e_{ij}(t) \cdot \exp(-\Delta t / \tau_e) + \text{STDP}_{ij}(t)$$

Weight updates occur only when a reward signal $r(t)$ arrives:

$$\Delta w_{ij} = \eta \cdot r(t) \cdot e_{ij}(t)$$

This implements the three-factor rule: Hebbian timing $\times$ eligibility $\times$ neuromodulatory reward.

*Source: `include/plasticity/stdp/nimcp_stdp.h` (field: `eligibility`)*

### 2.4 Homeostatic Plasticity

Three mechanisms maintain neural stability:

**Synaptic Scaling (Turrigiano 1998):**

$$w_{\text{scaled}} = w \times \left(\frac{r_{\text{target}}}{r_{\text{actual}}}\right)^\alpha$$

where $\alpha \in [0.5, 2.0]$ is the scaling exponent, $r_{\text{target}} = 5.0$ Hz (default).

**Intrinsic Plasticity (threshold adaptation):**

$$\frac{d\theta}{dt} = \frac{r_{\text{actual}} - r_{\text{target}}}{\tau_{\text{IP}}}$$

**Metaplasticity (BCM threshold sliding):**

$$\theta_m = \langle r^2 \rangle$$

The sliding threshold based on squared activity history.

*Source: `include/plasticity/homeostatic/nimcp_homeostatic.h`*

### 2.5 Backpropagation Through Time for SNN

SNN training uses BPTT with surrogate gradients. The non-differentiable Heaviside spike function $\Theta(V - V_{\text{thresh}})$ is replaced by a smooth surrogate:

$$\sigma'(V) = \frac{1}{\pi} \cdot \frac{1}{1 + (\pi \cdot \beta \cdot (V - V_{\text{thresh}}))^2}$$

The backward pass unrolls through time, computing temporal gradients of the loss with respect to synapse weights and firing thresholds.

*Source: `include/snn/nimcp_snn_training.h`, `src/snn/nimcp_snn_bptt.c`*

### 2.6 Adjoint Method for LNN

For the LNN ODE $\dot{x} = f(x, \theta, t)$, the adjoint state $a(t) = \partial L / \partial x(t)$ evolves backward:

$$\frac{da}{dt} = -a^\top \frac{\partial f}{\partial x}$$

Parameter gradients are accumulated during the backward sweep:

$$\frac{dL}{d\theta} = -\int_{t_1}^{t_0} a(t)^\top \frac{\partial f}{\partial \theta} \, dt$$

Adjoint truncation is capped at 50 backward steps. Gradients are normalized (not clipped) to target norm 1.0 when `use_gradient_normalization=true`, preventing the 100,000x reduction that caused mode collapse in earlier versions.

*Source: `src/lnn/nimcp_lnn_adjoint.c`*

### 2.7 AdamW Optimizer

Parameter updates with decoupled weight decay:

$$\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat{m}_t &= m_t / (1 - \beta_1^t), \quad \hat{v}_t = v_t / (1 - \beta_2^t) \\
\theta_t &= \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)
\end{aligned}$$

where $\lambda$ is the weight decay coefficient, decoupled from the adaptive learning rate.

*Source: `src/training/nimcp_training_optimizers.c`*

---

## 3. Fourier Neural Operators

### 3.1 Spectral Convolution

The core FNO operation learns a kernel in Fourier space:

$$(\mathcal{K}(\phi)v)(x) = \mathcal{F}^{-1}\bigl(R_\phi \cdot \mathcal{F}(v)\bigr)(x) + W_{\text{bypass}} \cdot v(x)$$

where $R_\phi \in \mathbb{C}^{c_{\text{out}} \times c_{\text{in}} \times k}$ are learned complex spectral weights truncated to $k$ Fourier modes, $\mathcal{F}$ denotes the FFT, and $W_{\text{bypass}}$ is a $1\times 1$ convolution residual path.

The spectral convolution is applied per-channel:

$$\hat{y}_{\text{out}}[j][m] = \sum_{i=0}^{c_{\text{in}}-1} (W_{\text{real}}[j,i,m] + i\,W_{\text{imag}}[j,i,m]) \cdot \hat{v}_{\text{in}}[i][m]$$

for modes $m = 0, \ldots, k-1$, followed by IFFT and GELU activation.

*Source: `include/training/nimcp_fno_layer.h`, `src/training/nimcp_fno_layer.c`*

### 3.2 FFT Implementation

NIMCP implements the Cooley-Tukey radix-2 FFT with $O(N \log N)$ complexity:

$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-2\pi i \, kn/N}, \qquad k = 0, \ldots, N-1$$

The inverse transform:

$$x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] \cdot e^{2\pi i \, kn/N}$$

Pre-computed twiddle factors $W_N^k = e^{-2\pi ik/N}$ and bit-reversal permutation tables are stored in the plan structure. Window functions reduce spectral leakage:

| Window | Formula |
|--------|---------|
| Hann | $w[n] = 0.5\bigl(1 - \cos(2\pi n / (N-1))\bigr)$ |
| Hamming | $w[n] = 0.54 - 0.46\cos(2\pi n / (N-1))$ |
| Blackman | $w[n] = 0.42 - 0.5\cos(2\pi n/(N-1)) + 0.08\cos(4\pi n/(N-1))$ |

Power spectral density: $P[k] = |X[k]|^2 = \text{Re}[k]^2 + \text{Im}[k]^2$

PSD in decibels: $P_{\text{dB}}[k] = 10 \log_{10}(P[k])$

Brain wave band power is computed by summing $P[k]$ over frequency bins corresponding to Delta (1--4 Hz), Theta (4--8 Hz), Alpha (8--13 Hz), Beta (13--30 Hz), and Gamma (30--100 Hz).

*Source: `include/utils/spectral/nimcp_fft.h`, `src/utils/spectral/nimcp_fft.c`*

### 3.3 FNO Layer Architecture

A complete FNO block consists of:

1. **Lifting**: Linear projection from input channels to hidden channels $W_{\text{lift}} \in \mathbb{R}^{h \times c_{\text{in}}}$
2. **Spectral blocks** ($n_{\text{blocks}}$, default 3): Each applies spectral convolution + bypass + GELU
3. **Projection**: Linear map from hidden channels to output $W_{\text{proj}} \in \mathbb{R}^{c_{\text{out}} \times h}$

For the audio processor, global average pooling reduces spatial dimensions before projection: $\bar{v}_c = \frac{1}{N} \sum_{n=0}^{N-1} v_c[n]$.

*Source: `include/training/nimcp_fno_layer.h`*

### 3.4 SNN-FNO Bridge

The FNO learns the population-level map $(V_t, I_{\text{syn}}) \mapsto (V_{t+\Delta t}, S_{t+\Delta t})$ from LIF ground truth, replacing per-neuron simulation when validation MSE falls below a threshold (default 0.01). The architecture per population:

- Lifting: $[h, n + n]$ (voltage + current)
- Spectral: $n_{\text{modes}} = n/4$ Fourier modes, $h = 16$ hidden channels
- Projection: $[2n, h]$ (next voltage + spike predictions)

Training uses a ring buffer of 256 state pairs, with online fine-tuning in validation mode.

*Source: `include/snn/nimcp_snn_fno.h`, `src/snn/nimcp_snn_fno.c`*

---

## 4. Quantum-Inspired Algorithms

### 4.1 Quantum Annealing

Optimization via simulated quantum tunneling through energy barriers:

**Classical acceptance (Metropolis):**

$$P_{\text{classical}} = \exp(-\Delta E / T)$$

**Quantum tunneling probability:**

$$P_{\text{tunnel}} = \Gamma \cdot \exp(-B / T^\alpha)$$

where $\Gamma$ is quantum strength (default 0.5), $B$ is the barrier height, and $T^\alpha$ the effective tunneling temperature.

**Combined acceptance:**

$$P_{\text{accept}} = \min\bigl(1, P_{\text{classical}} + P_{\text{tunnel}}\bigr)$$

**Cooling schedules:**

| Schedule | Formula |
|----------|---------|
| Exponential | $T(t) = T_0 \cdot \exp(-t/\tau)$ |
| Linear | $T(t) = T_0 - (T_0 - T_f) \cdot t / t_{\max}$ |
| Logarithmic | $T(t) = T_0 / \ln(1 + t)$ |
| Adaptive | Adjusted to maintain target acceptance rate (0.234) |

The adaptive variant uses Metropolis-Hastings with learned proposal covariance, adapting step sizes per dimension to achieve 2--5x faster convergence.

*Source: `include/optimization/quantum_annealing/nimcp_quantum_annealing.h`, `src/optimization/quantum_annealing/nimcp_quantum_annealing.c`*

### 4.2 Quantum Walk

A discrete-time quantum walk on the neural network graph provides $O(\sqrt{N})$ speedup for neuromodulator diffusion. The quantum state is:

$$|\psi\rangle = \sum_i \alpha_i |i\rangle, \qquad \sum_i |\alpha_i|^2 = 1$$

**Evolution operator:** $U = S \cdot C$ where:

- **Coin operator** $C$: Creates superposition at each node (Hadamard, Grover, or Fourier coin)
- **Shift operator** $S$: Propagates amplitude along edges

$$\alpha_j'' = \sum_{i \in \mathcal{N}(j)} \frac{\alpha_i'}{\sqrt{d_i}}$$

where $d_i$ is the degree of node $i$.

**Measurement (Born rule):** $P(i) = |\alpha_i|^2$

**Hybrid quantum-classical mixing:**

$$\alpha_{\text{final}} = (1 - \lambda) \cdot \alpha_{\text{quantum}} + \lambda \cdot \alpha_{\text{classical}}$$

**Decoherence** models environmental noise, gradually collapsing quantum superposition toward classical probabilities at rate $\gamma_{\text{decohere}} \in [0, 1]$.

*Source: `include/utils/quantum/nimcp_quantum_walk.h`, `src/utils/quantum/nimcp_quantum_walk.c`*

### 4.3 Quantum Monte Carlo

**Amplitude estimation** via importance sampling: Estimate $|\langle \phi | \psi \rangle|^2$ using proposal distribution $q$:

$$\hat{P} = \frac{1}{N} \sum_{i=1}^{N} \frac{|\psi(x_i)|^2}{q(x_i)}, \qquad x_i \sim q$$

**Finite-shot measurement** simulates realistic quantum hardware by sampling from the multinomial distribution $\text{Multi}(N_{\text{shots}}; p_0, p_1, \ldots)$ where $p_i = |\alpha_i|^2$.

**Partition function estimation:**

$$Z = \sum_i \exp(-E_i / T), \qquad F = -T \ln Z, \qquad C = \text{Var}(E) / T^2$$

using MCMC with burn-in, thinning, and thermodynamic integration.

**Entropy estimation (Renyi family):**

$$\begin{aligned}
H_{\text{Shannon}} &= -\sum_i p_i \ln p_i \\
H_2 &= -\ln\!\left(\sum_i p_i^2\right) \\
H_\infty &= -\ln\!\left(\max_i p_i\right)
\end{aligned}$$

Estimated via MC sampling with stratified variance reduction.

*Source: `include/utils/quantum/nimcp_quantum_monte_carlo.h`, `src/utils/quantum/nimcp_quantum_monte_carlo.c`*

### 4.4 Quantum Shannon Integration

The quantum-Shannon system combines quantum walk propagation with Shannon information monitoring:

**Channel capacity:**

$$C = B \cdot \log_2(1 + \text{SNR}) \quad \text{bits/second}$$

**Propagation efficiency:**

$$\eta = \frac{I(\text{source}; \text{targets})}{H(\text{source})}$$

Bottleneck synapses are identified when capacity utilization falls below threshold. The system adaptively routes around bottlenecks by biasing the quantum coin operator toward high-capacity neighbors.

*Source: `include/utils/quantum/nimcp_quantum_shannon.h`*

### 4.5 Quantum-Classical Bridge

Quantum algorithm results feed into neural updates via:

1. **Quantum annealing** optimizes synaptic weight matrices by periodically escaping local minima during plasticity steps
2. **Quantum walk** distributions become neuromodulator concentration fields (dopamine, serotonin, ACh, NE)
3. **QMC amplitude estimation** guides importance-weighted sampling in episodic replay
4. **Shannon feedback** drives adaptive routing of information through the network

---

## 5. Information Theory

### 5.1 Shannon Entropy

$$H(X) = -\sum_{i} p_i \ln p_i$$

Computed with safe logarithm: $\ln(\max(x, 10^{-10}))$ to prevent $-\infty$. Used for monitoring neural activity diversity, attention entropy, and network health.

*Source: `include/utils/math/nimcp_math_helpers.h` (`nimcp_entropy`)*

### 5.2 Mutual Information and KL Divergence

$$I(X; Y) = H(X) - H(X|Y) = \sum_{x,y} p(x,y) \ln \frac{p(x,y)}{p(x)p(y)}$$

**KL Divergence:**

$$D_{\text{KL}}(P \| Q) = \sum_i p_i \ln \frac{p_i}{q_i}$$

Used for loss functions, distribution comparison, and bottleneck detection.

*Source: `src/utils/quantum/nimcp_quantum_monte_carlo.c` (`qmc_kl_divergence`), `include/physics/geometry/nimcp_information_geometry.h`*

### 5.3 Integrated Information (IIT Phi)

$$\Phi = \min_{\text{partitions}} \bigl[ I(\text{whole}) - \sum I(\text{parts}) \bigr]$$

Measures irreducible causal structure. Exponential in system size; approximations used for large networks.

*Source: `include/utils/statistics/nimcp_information_theory.h`*

### 5.4 Partial Information Decomposition (PID)

For two sources $S_1, S_2$ predicting target $T$:

$$I(S_1, S_2 ; T) = U_1 + U_2 + R + S$$

where $U_i$ is unique information from source $i$, $R$ is redundancy, and $S$ is synergy. Multiple decomposition methods are supported: BROJA, CCS, MMI, DEP, and iterative BROJA.

*Source: `include/utils/statistics/nimcp_information_theory.h`*

### 5.5 Fisher Information Matrix

$$F_{ij}(\theta) = \mathbb{E}\!\left[\frac{\partial \ln p(x|\theta)}{\partial \theta_i} \cdot \frac{\partial \ln p(x|\theta)}{\partial \theta_j}\right]$$

The empirical Fisher is estimated from gradient samples. Regularized inversion: $\tilde{F} = F + \lambda I$ with $\lambda = 10^{-6}$.

*Source: `include/physics/geometry/nimcp_information_geometry.h` (`nimcp_fisher_compute`)*

### 5.6 Information Geometry

The parameter space of neural network weights forms a Riemannian manifold with the Fisher information matrix as its metric tensor.

**Natural gradient:**

$$\tilde{\nabla} L = F^{-1} \nabla L$$

converges 2--10x faster than standard SGD by accounting for the geometry of the parameter space. Gradient clipping at norm 10.0, with optional momentum and warmup scheduling.

**Geodesic distance** between parameter configurations $\theta_a, \theta_b$ is computed on the Fisher manifold:

$$d(\theta_a, \theta_b) = \inf_\gamma \int_0^1 \sqrt{\dot\gamma(t)^\top F(\gamma(t)) \, \dot\gamma(t)} \, dt$$

**Ricci curvature** measures how volume changes along geodesics, informing adaptive learning rate schedules.

*Source: `include/physics/geometry/nimcp_information_geometry.h`*

---

## 6. Positional Encoding

### 6.1 Sinusoidal Encoding (Vaswani 2017)

$$\begin{aligned}
\text{PE}(\text{pos}, 2i) &= \sin\!\left(\frac{\text{pos}}{b^{2i/d}}\right) \\
\text{PE}(\text{pos}, 2i+1) &= \cos\!\left(\frac{\text{pos}}{b^{2i/d}}\right)
\end{aligned}$$

where $b = 10000$ (configurable via `frequency_base`), $d$ is the embedding dimension, and $i$ indexes the dimension pair. Fixed, requires no training, and extrapolates to unseen positions.

### 6.2 Rotary Position Embedding (RoPE)

Encodes relative position via 2D rotation applied to query-key pairs:

$$\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

where $m$ is position, $\theta_i = b^{-2i/d}$, and $b = 10000$ (default `rope_base`). The dot product $\langle q'_m, k'_n \rangle$ depends only on relative position $m - n$. NTK-aware scaling extends context length by adjusting the base: $b' = b \cdot s^{d/(d-2)}$.

### 6.3 ALiBi (Attention with Linear Biases)

No explicit positional encoding. Instead, a linear bias is added to attention scores:

$$\text{bias}[h][i][j] = -m_h \cdot |i - j|$$

Slopes follow a geometric sequence:

$$m_h = 2^{-8(h+1)/H}$$

where $H$ is the number of attention heads. This provides the simplest and most efficient long-sequence support.

### 6.4 Relative Position Representations (Shaw 2018)

Separate key and value embeddings indexed by clipped relative position $\text{clip}(i - j, -k, k)$ where $k$ is `max_relative_pos`.

*Source: `include/utils/encoding/nimcp_positional_encoding.h`*

---

## 7. Signal Processing

### 7.1 Fast Fourier Transform

The Cooley-Tukey radix-2 algorithm with $O(N \log N)$ complexity for power-of-2 sizes (2--65536). Real-to-complex FFT produces $N/2 + 1$ frequency bins. Frequency resolution: $\Delta f = f_s / N$.

### 7.2 Spectral Analysis

**Magnitude spectrum:** $|X[k]| = \sqrt{\text{Re}[k]^2 + \text{Im}[k]^2}$

**Phase spectrum:** $\phi[k] = \text{atan2}(\text{Im}[k], \text{Re}[k])$

**Band power:** $P_{\text{band}} = \sum_{k: f_k \in [f_{\text{lo}}, f_{\text{hi}}]} |X[k]|^2$

Bin-to-frequency conversion: $f_k = k \cdot f_s / N$.

*Source: `include/utils/spectral/nimcp_fft.h`*

---

## 8. Optimization

### 8.1 Gradient Descent Variants

NIMCP implements per-network learning rates, layer-wise LR scaling (tau parameters at 0.1x, bias at 2.0x for LNN), and gradient normalization that always normalizes to target norm rather than clipping. The default gradient clip is 100.0 (raised from 1.0 after mode collapse diagnosis).

### 8.2 Natural Gradient

$$\theta_{t+1} = \theta_t - \eta \, F^{-1}(\theta_t) \, \nabla L(\theta_t)$$

See Section 5.6 for the Fisher information matrix computation. Adaptive damping adjusts the regularization parameter based on the ratio of actual to predicted loss reduction.

*Source: `include/physics/geometry/nimcp_information_geometry.h`*

### 8.3 Quantum Annealing for Weight Optimization

Periodically invoked during plasticity (every $K$ learning steps) to escape local minima in the full synaptic weight matrix. See Section 4.1.

### 8.4 Elastic Weight Consolidation

$$\mathcal{L}(\theta) = \mathcal{L}_B(\theta) + \sum_i \frac{\lambda}{2} F_i \left(\theta_i - \theta_{A,i}^*\right)^2$$

where $F_i$ is the diagonal of the Fisher information matrix computed on task $A$, $\theta_{A,i}^*$ are the optimal parameters for task $A$, and $\lambda$ controls the consolidation strength. Prevents catastrophic forgetting by penalizing changes to parameters that were important for previous tasks.

### 8.5 Federated Averaging (Swarm Learning)

$$w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} w_{t+1}^k$$

where $n_k$ is the number of samples on edge node $k$ and $n = \sum_k n_k$. Used in NIMCP's swarm runtime for gossip-based model aggregation with Byzantine fault tolerance.

*Source: `src/swarm/nimcp_swarm_sync.c`*

---

## 9. Geometry and Manifolds

### 9.1 Riemannian Metrics

The metric tensor on a 2D surface manifold:

$$\gamma_{ab} = \frac{\partial X}{\partial \sigma^a} \cdot \frac{\partial X}{\partial \sigma^b}$$

where $\sigma = (\sigma^0, \sigma^1)$ are local coordinates (longitudinal and azimuthal).

For a cylinder of radius $r$: $\gamma = \begin{pmatrix} 1 & 0 \\ 0 & r^2 \end{pmatrix}$

### 9.2 Surface Area and Nambu-Goto Action

The total surface area of the manifold $M(G)$ assigned to graph $G$:

$$S_{M}(G) = \sum_i \int \sqrt{\det(\gamma_i)} \, d\sigma^0 \, d\sigma^1$$

This is formally identical to the Nambu-Goto action from string theory (with string tension $T = 1$). Computed via Gaussian quadrature with 8--64 points, adaptively chosen based on local curvature.

**Strebel's theorem:** In the absence of boundary conditions, the minimal surface is exactly cylindrical. With boundary conditions, quadratic differentials give the minimal surface.

### 9.3 Geometric Parameters

$$\chi = w/r \quad \text{(aspect ratio)}, \qquad \rho = w'/w \quad \text{(branching ratio)}, \qquad \lambda = l/w \quad \text{(length ratio)}$$

Branch point topology is mapped to Feynman diagram vertices via pants decomposition.

*Source: `include/core/geometry/nimcp_surface_manifold.h`*

### 9.4 Neural Manifold Analysis

The intrinsic dimensionality of neural activity is estimated from samples on the ambient-dimensional activity space. Local curvature at a point is computed for adaptive learning rate scheduling. Geodesic computation between points on the manifold uses iterative path optimization.

*Source: `include/physics/geometry/nimcp_information_geometry.h` (`nimcp_manifold_*`)*

---

## 10. Probability and Statistics

### 10.1 Sigmoid and Softmax

**Sigmoid:** $\sigma(x) = 1 / (1 + e^{-x})$

**Safe exponential:** Input clamped to $[-88, 88]$ before $\exp$ to prevent overflow ($e^{89} = \infty$ in single precision).

**Softmax with temperature:**

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

*Source: `include/utils/math/nimcp_math_helpers.h`*

### 10.2 Exponential Moving Average

$$\text{EMA}_t = \alpha \cdot x_t + (1 - \alpha) \cdot \text{EMA}_{t-1}$$

EMA values are guarded against NaN/Inf corruption: `NIMCP_EMA_GUARD(ema_var, fallback)` resets to fallback value if non-finite.

*Source: `include/utils/math/nimcp_math_helpers.h`*

### 10.3 Cosine Similarity (Diversity Loss)

Anti-mode-collapse uses a ring buffer of 16 recent outputs. The diversity penalty:

$$\mathcal{L}_{\text{div}} = \frac{w_{\text{div}}}{|B|(|B|-1)} \sum_{i \neq j} \frac{y_i \cdot y_j}{\|y_i\| \|y_j\|}$$

with $w_{\text{div}} = 0.1$, penalizing high cosine similarity between outputs.

### 10.4 Quantum State Fidelity

$$F(|\psi_1\rangle, |\psi_2\rangle) = |\langle \psi_1 | \psi_2 \rangle|^2 = \left(\sum_i a_i^{(1)} a_i^{(2)}\right)^2$$

for real amplitude vectors.

*Source: `include/utils/quantum/nimcp_quantum_monte_carlo.h` (`qmc_fidelity`)*

---

## 11. Neuromodulation

### 11.1 Dopamine Reward Prediction Error

$$\delta_{\text{DA}} = r(t) - \hat{V}(s_t)$$

Phasic bursts ($\delta > 0$) amplify STDP learning by factor $m_{\text{burst}} = 3.0$. Tonic dopamine sets baseline learning rate. Diffusion modeled via quantum walk (Section 4.2).

### 11.2 Serotonin, Norepinephrine, Acetylcholine

Six neuromodulators modulate plasticity and network dynamics. Each has a spatial concentration field $[M](x, t)$ that decays exponentially and is replenished by source nuclei. The `neuromodulator_system_t` (already a pointer typedef) manages diffusion, reuptake, and receptor binding across 33+ brain regions.

*Source: `include/core/neuromodulators/nimcp_neuromodulators.h`*

---

## 12. Memory Mathematics

### 12.1 Engram Encoding

Pattern completion via associative recall. An engram $e$ is a stored activation pattern; retrieval uses cosine similarity between cue and stored patterns, selecting the engram with highest overlap.

### 12.2 Episodic Replay

Importance-weighted sampling selects experiences for replay:

$$P(\text{replay}_i) \propto |\delta_i|^\alpha + \epsilon$$

where $\delta_i$ is the TD error at encoding time, $\alpha$ controls prioritization.

### 12.3 Memory Consolidation

Synaptic weight decay during consolidation:

$$w(t) = w_0 \cdot \exp(-t / \tau_{\text{decay}}) + w_{\text{consolidated}} \cdot (1 - \exp(-t / \tau_{\text{consolidate}}))$$

### 12.4 Habituation

Repeated stimulus exposure produces exponential dampening:

$$h(t) = h_0 \cdot \exp(-n_{\text{exposures}} / \tau_h)$$

---

## 13. Safety Mathematics

### 13.1 CRC32 Checksum

Each audit log entry carries a CRC32 checksum computed over its payload. Mismatches indicate tampering. Monotonic sequence numbers detect deletion (gaps indicate removed entries).

*Source: `include/security/nimcp_audit_log.h`*

### 13.2 Byzantine Gradient Detection

In swarm learning, gradients from edge nodes are validated using EMA-tracked statistics:

$$z_k = \frac{\|g_k\| - \mu_{\text{EMA}}}{\sigma_{\text{EMA}}}$$

Nodes with $|z_k| > z_{\text{thresh}}$ are flagged as potentially Byzantine and excluded from federated averaging.

*Source: `src/swarm/nimcp_swarm_byzantine.c`*

### 13.3 LGSS Safety Lattice

The Layered Governance Safety System evaluates actions against a partial order of safety levels. An action at level $\ell$ requires clearance $\geq \ell$:

$$\text{allow}(a) \iff \text{clearance}(a) \geq \text{level}(a) \quad \forall \text{ safety domains}$$

Five pipeline points (input validation, action interception, motor gate, training guard, reward alignment) form a monotonic safety chain: blocking at any point propagates to all downstream stages.

*Source: `include/security/lgss/nimcp_lgss.h`*

---

## References

1. Bi, G. & Poo, M. (1998). Synaptic modifications in cultured hippocampal neurons. *J. Neurosci.*, 18(24), 10464--10472.
2. Bienenstock, E., Cooper, L., & Munro, P. (1982). Theory for the development of neuron selectivity. *J. Neurosci.*, 2(1), 32--48.
3. Turrigiano, G. et al. (1998). Activity-dependent scaling of quantal amplitude in neocortical neurons. *Nature*, 391, 892--896.
4. Greydanus, S. et al. (2019). Hamiltonian Neural Networks. *NeurIPS 2019*.
5. Li, Z. et al. (2021). Fourier Neural Operator for parametric partial differential equations. *ICLR 2021*.
6. Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS 2017*.
7. Su, J. et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. *arXiv:2104.09864*.
8. Press, O. et al. (2022). Train Short, Test Long: Attention with Linear Biases. *ICLR 2022*.
9. Kadowaki, T. & Nishimori, H. (1998). Quantum annealing in the transverse Ising model. *Phys. Rev. E*, 58(5), 5355.
10. Amari, S. (1998). Natural gradient works efficiently in learning. *Neural Computation*, 10(2), 251--276.
11. Kirkpatrick, J. et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13), 3521--3526.
12. Tononi, G. (2004). An information integration theory of consciousness. *BMC Neuroscience*, 5, 42.
13. Williams, P. & Beer, R. (2010). Nonnegative decomposition of multivariate information. *arXiv:1004.2515*.

---

*Generated from NIMCP v2.6.4 source code. All equations correspond to implemented functions in the codebase.*

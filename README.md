# State Space Models for Transport PDEs: Out-of-Distribution Benchmarks

This repository evaluates State Space Models (SSMs) like Mamba alongside baseline neural operators (FNO, Laplace Neural Operators, WNO, Oformer, Galerkin Operator Transformers) for solving continuous dynamic transport Partial Differential Equations (PDEs). The models are systematically tested under nominal and extreme/out-of-distribution (OOD) initial condition regimes to evaluate phase errors, numerical dissipation, steep gradients, and non-linear coupling.

---

## 🔬 Benchmark Dynamical Systems

### 1. Kuramoto-Sivashinsky (KS) Equation
$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + \frac{\partial^2 u}{\partial x^2} + \frac{\partial^4 u}{\partial x^4} = 0$$
- **Physical Phenomenon:** Spatiotemporal chaos, turbulence, and phase dynamics
- **Biomedical Application:** Cardiac electrical wave turbulence and neural bursting patterns.
- **Initial Conditions:**
  * **Nominal Regime (92% Train / 75% Test):** Modulated Cosine Wave  
    $$u_0(x) = \text{amp} \cdot \cos(x + \phi)(1.0 + \sin(x + \phi)), \quad \text{amp} \sim \mathcal{U}(0.15, 0.25), \ \phi \sim \mathcal{U}(0, 2\pi)$$
  * **Extreme / OOD Regime (8% Train / 25% Test):** Multi-Frequency Harmonic Mixture  
    $$u_0(x) = a_1 \sin(2.0x + \phi) + a_2 \cos(3.0x - \phi), \quad a_1 \sim \mathcal{U}(0.15, 0.25), \ a_2 \sim \mathcal{U}(0.08, 0.15), \ \phi \sim \mathcal{U}(0, 2\pi)$$

---

### 2. Viscous Burgers' Equation
$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$
- **Physical Phenomenon:** Nonlinear advection and shock dissipation.
- **Biomedical Application:** Arterial pulse wave steepening and blood flow shock profiles.
- **Initial Conditions & Viscosity ($\nu$):**
  * **Nominal Regime (92% Train / 75% Test):** Standard Smooth Shock Formation  
    $$u_0(x) = \sin(x) + 0.5 \sin(2.0x), \quad \nu \sim \mathcal{U}(0.01, 0.03)$$
  * **Extreme / OOD Regime (8% Train / 25% Test):** Aggressive Low-Viscosity Shocks  
    $$u_0(x) = -\sin(x) \cdot 1.5, \quad \nu \sim \mathcal{U}(0.002, 0.008)$$

---

### 3. Korteweg–de Vries (KdV) Equation
$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + \frac{\partial^3 u}{\partial x^3} = 0$$
- **Physical Phenomenon:** Soliton wave dynamics and dispersion.
- **Biomedical Application:** Nerve impulse propagation and lossless action potentials along axons.
- **Initial Conditions & Dispersion ($\delta^2$):**
  * **Nominal Regime (92% Train / 75% Test):** Single Soliton Wave  
    $$u_0(x) = 3.0 c_1 \text{cosh}^{-2}\left(0.5 \sqrt{c_1} x\right), \quad c_1 \sim \mathcal{U}(0.8, 1.2), \ \delta^2 \sim \mathcal{U}(0.001, 0.003)$$
  * **Extreme / OOD Regime (8% Train / 25% Test):** Multi-Soliton Interaction  
    $$u_0(x) = 3.0 \text{cosh}^{-2}(0.5x) + 2.0 \text{cosh}^{-2}(0.4(x - \pi)), \quad \delta^2 \sim \mathcal{U}(0.004, 0.008)$$

---

### 4. Advection-Diffusion Equation
$$\frac{\partial u}{\partial t} + c \frac{\partial u}{\partial x} = D \frac{\partial^2 u}{\partial x^2}$$
- **Physical Phenomenon:** Bulk transport and physical dispersion.
- **Biomedical Application:** Drug delivery kinetics and oxygen dispersion across capillary beds.
- **Initial Conditions, Velocity ($v$), and Diffusion ($D$):**
  * **Nominal Regime (92% Train / 75% Test):** Harmonic Trigonometric Mixture  
    $$u_0(x) = \sin(x) + 0.5 \cos(2.0x), \quad v \sim \mathcal{U}(0.5, 1.5), \ D \sim \mathcal{U}(0.02, 0.08)$$
  * **Extreme / OOD Regime (8% Train / 25% Test):** Gaussian-Modulated Steep Wave  
    $$u_0(x) = \sin(3.0x) \cdot \exp\left(-\frac{(x-\pi)^2}{2.0}\right), \quad v \sim \mathcal{U}(1.8, 3.0), \ D \sim \mathcal{U}(0.005, 0.02)$$

---

## 📂 Code Structure & Included Notebooks

- `mamba-mamba-work-correct.ipynb` — Primary State Space Mamba model implementation and OOD evaluation on transport PDEs.
- `fno-mamba-work.ipynb` — Fourier Neural Operator (FNO) integration and baseline comparison.
- `lno_all_systems.ipynb` — Laplace Neural Operator (LNO) benchmark across all four transport systems.
- `wno-mamba-work.ipynb` — Wavelet Neural Operator (WNO) performance under multiscale OOD conditions.
- `oformer_(_mamba_work).ipynb` — Original Operator Transformer architecture setup.
- `G_oformer_(_mamba_work).ipynb` — Galerkin Operator Transformer (G-Oformer) comparative experiments.

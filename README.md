# Skrample 0.6.1
Composable sampling functions for diffusion models

## Status
Production-tested on all popular diffusion models. The library has significantly matured since 0.5

### Quickstart
Fastest way to jump in is [examples](./examples/). The classes and functions themselves have docstrings and type hints, so it's recommended to make liberal use of your IDE or python `help()`

### Feature Flags
 - `beta-schedule` -> `scipy` : For the `Beta()` schedule modifier
 - `brownian-noise` -> `torchsde` : For the `Brownian()` noise generator
 - `cdf-schedule` -> `scipy` : For the `Probit()` schedule
 - `diffusers-wrapper` -> `torch` : For the `diffusers` integration module
 - `pytorch` -> `torch` : For the `pytorch` module
   - `pytorch.noise` : Custom generators
 - `all` : All of the above
 - `dev` : For running `tests/`

### Structured Samplers
These samplers are written inside-out to be compatible with Diffusers and similar frameworks

- Euler
  - Stochastic
- DPM
  - Order 1-3
  - Stochastic
- Adams/IPNDM
  - Order 1-9
  - Stochastic
- UniP & UniPC
  - Order 1-9
  - Stochastic
  - Custom predictor via other SkrampleSampler types
- SPC
  - Basic fully customizable midpoint corrector

### Functional Samplers
These samplers are written using closures similar to ksampler

- RKUltra
  - Arbitrary Runge-Kutta solver
  - Order 1-15, customizable through tableaux system
  - Stochastic
- DynasauRK
  - Procedural Runge-Kutta solver
  - Order 2-4
  - Stochastic
- RKMoire
  - Experimental
  - Embedded Runge-Kutta solver
  - Order 2-6, customizable through tableaux system

### Schedules
- Linear
  - Flow-matching default
- Scaled
  - Variance-preserving default
- ZSNR

### Subschedules
Replaces sigmas on an existing schedule

- Karras
- Exponential
- Beta
- Probit

### Schedule Modifiers
Modifies timestep spacing of a schedule

- FlowShift
- Hyper
- Sinner

### Models
- Data / Sample / X-Pred
- Noise / Epsilon / Ε-Pred
- Velocity / V-Pred
- Flow / U-pred

### Noise generators
- Random
- Brownian
- Offset
- Pyramid

## Integrations
### Diffusers
- [X] Compatibile with DiffusionPipeline
- [X] Import from config
  - [X] Sampler
  - [X] Schedule
  - [X] Predictor
- [X] Structured sampler wrapper
- [X] Functional sampler wrappers
  - [X] RKUltra
  - [X] DynasauRK

## Implementations
### quickdif
My diffusers cli [quickdif](https://github.com/Beinsezii/quickdif) has full support for all major Diffusers-compatible Skrample features, allowing extremely fine-grained customization.

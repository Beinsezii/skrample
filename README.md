# `skrample`
Composable sampling functions for diffusion models

## Status
Vertical slice, gradually overtaking many diffusers features in [quickdif](https://github.com/Beinsezii/quickdif.git)

### Feature Flags
 - `beta-schedule` -> `scipy` : For the `Beta()` schedule modifier
 - `brownian-noise` -> `torchsde` : For the `Brownian()` noise generator
 - `diffusers-wrapper` -> `torch` : For the `diffusers` integration module
 - `pytorch` -> `torch` : For the `pytorch` module
   - `pytorch.noise` : Custom generators
 - `all` : All of the above
 - `dev` : For running `tests/`

### Samplers
- Euler
  - Ancestral
- DPM
  - 1st order, 2nd order, 3rd order
  - SDE
- IPNDM
  - Ancestral (from Euler)
- UniPC
  - N order, limited to 9 for stability
  - Custom solver via other SkrampleSampler types

### Schedules
- Linear
- Scaled
  - `uniform` flag, AKA `"trailing"` in diffusers
- Flow
  - Dynamic and non-dynamic shifting
- ZSNR

### Schedule modifiers
- Karras
- Exponential
- Beta

### Predictors
- Epsilon
- Velocity / vpred
- Flow

### Noise generators
- Random
- Brownian
- Offset
- Pyramid

## Integrations
### Diffusers
- [X] Compatibility for pipelines
  - [X] SD1
  - [X] SDXL
  - [X] SD3
  - [X] Flux
  - [ ] Others?
- [X] Import from config
  - [ ] Sampler
    - Not sure this is even worthwhile. All Skrample samplers work everywhere
  - [X] Schedule
  - [X] Predictor
- [X] Manage state
  - [X] Steps
  - [X] Higher order
  - [X] Generators
  - [X] Config as presented

## Implementations
### quickdif
A basic test bed is available for https://github.com/Beinsezii/quickdif.git on the `skrample` branch

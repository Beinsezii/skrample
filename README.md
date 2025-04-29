# Skrample 0.2.2
Composable sampling functions for diffusion models

## Status
Vertical slice, overtaking most diffusers features in [quickdif](https://github.com/Beinsezii/quickdif.git)

### Feature Flags
 - `beta-schedule` -> `scipy` : For the `Beta()` schedule modifier
 - `brownian-noise` -> `torchsde` : For the `Brownian()` noise generator
 - `cdf-schedule` -> `scipy` : For the `SigmoidCDF()` schedule
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
- SigmaCDF
- ZSNR

### Schedule modifiers
- Karras
- Exponential
- FlowShift
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
  - [X] Sampler
  - [X] Schedule
  - [X] Predictor
- [X] Manage state
  - [X] Steps
  - [X] Higher order
  - [X] Generators
  - [X] Config as presented

## Implementations
### quickdif
My diffusers cli [quickdif](https://github.com/Beinsezii/quickdif) has full support for all major Skrample features, allowing extremely fine-grained customization.

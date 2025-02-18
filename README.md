# `skrample`
Composable sampling functions for diffusion models

## Status
Early alpha, contains only a few features and a Diffusers wrapper class.

### Feature Flags
 - `beta-schedule` : For the `Beta()` schedule modifier
 - `diffusers-wrapper` : For the `diffusers` integration module
 - `all` : All of the above
 - `dev` : For running `tests/`

### Samplers
- Euler
- DPM
  - 1st order, 2nd order
  - SDE
- UniPC
  - N order, limited to 9 for stability
  - Custom solver via other SkrampleSampler types

### Schedules
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

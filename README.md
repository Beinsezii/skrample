# `skrample`
Composable sampling functions for diffusion models

## Status
Early alpha, contains only a few features and a Diffusers wrapper class.

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
- [ ] Import from config
  - [ ] Sampler
  - [ ] Schedule
  - [ ] Predictor
- [ ] Manage state
  - [X] Steps
  - [X] Higher order
  - [X] Generators
  - [ ] Config as presented

## Implementations
### quickdif
A basic test bed is available for https://github.com/Beinsezii/quickdif.git on the `skrample` branch

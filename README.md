# `skrample`
Composable sampling functions for diffusion models

## Status
Early alpha, contains only a few features and a Diffusers wrapper class.

### Samplers
- Euler
- DPM
  - 1st order, 2nd order
- UniPC
  - N order, limited to 9 for stability
  - Custom solver via other SkrampleSampler types

### Schedules
- Scaled
  - `uniform` flag, AKA `"trailing"` in diffusers
- Flow
 - Dynamic and non-dynamic shifting

### Predictors
- Epsilon
- Velocity / vpred
- Flow

## Integrations
### Diffusers
- [X] Compatibility for pipelines
  - [X] SD1
  - [X] SDXL
  - [ ] SD3
  - [X] Flux
  - [ ] Others?
- [ ] Import from config
  - [ ] Sampler
  - [ ] Schedule
  - [ ] Predictor
- [ ] Manage state
  - [X] Steps
  - [X] Higher order
  - [ ] Generators

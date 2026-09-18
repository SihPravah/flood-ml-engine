# PRAVAHA ML Engine

ML and hydrology service for PRAVAHA.

## Responsibilities

- Consume canonical PRAVAHA data-contract payloads
- Build hydrological and temporal features
- Estimate runoff and catchment response
- Predict flash-flood risk
- Estimate confidence and lead time
- Expose explainable prediction outputs

## Development Rules

- Branch from `dev`
- Never push directly to `main`
- Feature work returns to `dev` through Pull Requests
- Do not modify cross-repository data contracts locally
- Canonical data structure is defined by `DATA_CONTRACT.md` in `flood-data-iot`
- Missing dependencies must be mocked while preserving the final contract

## Python

Python 3.11.x

## Static GIS Context

The live ML boundary remains `FusedCatchmentState v2.1`. For the focused
Chandrabani PS 26192 demo, static terrain/road/stream context is loaded through
the ML static-context adapter and must retain source-status metadata such as
`OPEN_REAL_DATA`, `DERIVED_FROM_REAL_DATA`, `ESTIMATED`, `DEMO`, and
`NOT_AVAILABLE`.

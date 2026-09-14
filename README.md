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
# PRAVAHA Engineering Constitution

## MISSION
PRAVAHA is a software-only hyper-local flash-flood prediction and anticipatory disaster decision-support system for hilly regions.

It must align with the SIH problem statement and use:
- rainfall
- soil moisture
- terrain / DEM
- slope stability
- historical landslide inventories
- real-time or simulated IoT
- hydrology
- village/ward-level warning
- evacuation support

## FOUR-REPO ARCHITECTURE
- flood-data-iot
- flood-ml-engine
- flood-backend
- flood-frontend

## BRANCH POLICY
- main is protected release
- dev is integration
- feature/codex branches come from dev
- no direct pushes to main

## CONTRACT AUTHORITY
flood-data-iot/DATA_CONTRACT.md is the cross-repo source of truth.

Any shared API/schema/unit/enum/ID change must emit:

ARCHITECTURE / CONTRACT CHANGE ALERT

## SEMANTIC INVARIANTS
- risk != confidence
- prediction != observation
- prediction != authority confirmation
- 0 != missing
- SIMULATED != OBSERVED
- ESTIMATED != OBSERVED
- intensity != accumulated rainfall
- low risk + low confidence != safe
- AVOID != CLOSED

## PROVENANCE
- OBSERVED
- DERIVED
- ESTIMATED
- SIMULATED
- MISSING

## RISK LEVELS
- LOW
- WATCH
- WARNING
- HIGH
- SEVERE

## UNITS
- rainfall intensity: mm/hr
- accumulated rainfall: mm
- discharge: m^3/s
- elevation/distance: metres
- slope: fraction unless contract says otherwise
- probability/confidence: 0..1
- GeoJSON API coordinates: [longitude, latitude]

## SAFETY
- PRAVAHA is decision support
- never claim guaranteed safe route
- official closures and evacuation orders must remain distinguishable from predictions
- if data is insufficient, surface uncertainty instead of declaring safety
- synthetic/demo metrics are not operational validation

## DEVELOPMENT VS OPERATIONAL DATA
- SIMULATED data is permitted for development, testing and SIH demonstrations
- simulated evidence must remain visibly tagged as SIMULATED
- simulated evidence must never be presented as a real observation
- operational-mode decisions must not silently depend on simulated evidence
- demo scenarios must be clearly identifiable as demonstrations

## PRODUCT INTELLIGENCE
Target chain:

multi-source observations
-> normalization
-> temporal fusion
-> hydrology
-> flash-flood risk
-> confidence
-> landslide susceptibility
-> trend/threshold anticipation
-> hazard cascades
-> drainage overload
-> road flood exposure
-> risk-aware routing
-> ward/village impact
-> evacuation readiness
-> city intelligence
-> premium map interface

## FRONTEND PRINCIPLES
- map-first
- premium modern navigation UX
- click-anything detail drawer
- risk + confidence + reasons + provenance + freshness
- flood, landslide, drainage, road, sensor, ward, shelter and route layers
- no fake live data
- no generic hackathon dashboard

## ENGINEERING RULES
Before coding:
- inspect existing implementation
- inspect tests
- inspect contracts
- preserve working architecture

After coding:
- run tests/builds
- fix failures
- add tests
- check git diff
- document important changes
- never claim success without verification

Do not rewrite working modules unnecessarily.

## MULTI-REPOSITORY SAFETY
- modify only the repository explicitly authorized by the current task
- do not make opportunistic changes in sibling repositories
- cross-repository changes require explicit task authorization
- never silently change DATA_CONTRACT.md to make an implementation fit
- when a required shared-contract change is discovered, stop that shared change and emit an ARCHITECTURE / CONTRACT CHANGE ALERT
- independent work that does not require the contract change may continue

## DEFINITION OF DONE
A task is not complete until:
- code works
- tests/build pass
- contracts remain compatible
- safety semantics are preserved
- docs are updated
- demo/simulated evidence is clearly marked
- changes are Git-ready

## REPO-SPECIFIC RESPONSIBILITIES

### flood-data-iot
"What do we currently know?"

Owns adapters, normalization, temporal history, fusion, provenance and canonical FusedCatchmentState.
Does not make flood predictions.

### flood-ml
"What does the current state imply and what may happen next?"

Owns hydrology, ML, confidence, landslide, anticipation, cascade, drainage, road risk, routing, impact, evacuation and city intelligence.

### flood-backend
"How is intelligence orchestrated and exposed?"

Owns APIs, service integration, caching/state, map endpoints, routing endpoints, alerts and frontend-facing schemas.
Must not duplicate ML logic.

### flood-frontend
"How does a human understand and act on the intelligence?"

Owns premium map-first UX, layers, drawers, route comparison, timeline, alerts and demo presentation.

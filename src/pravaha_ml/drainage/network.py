from dataclasses import dataclass
from typing import Iterable

from pravaha_ml.drainage.capacity import (
    calculate_rectangular_manning_capacity,
)
from pravaha_ml.drainage.flow import (
    CatchmentRunoffInput,
    calculate_catchment_discharge,
)
from pravaha_ml.drainage.models import (
    DrainDynamicLoad,
    DrainRiskAssessment,
    DrainStaticProfile,
)
from pravaha_ml.drainage.risk import (
    assess_drain_overflow_risk,
)


@dataclass(frozen=True)
class DrainConnection:
    """
    Directed drainage-network connection.

    Water flows:

        upstream_drain_id
                ↓
        downstream_drain_id

    flow_fraction specifies the fraction of conveyed upstream
    discharge entering this downstream branch.

    A drain may therefore split flow across multiple downstream
    connections.
    """

    upstream_drain_id: str
    downstream_drain_id: str

    flow_fraction: float = 1.0

    def __post_init__(self) -> None:
        if not self.upstream_drain_id.strip():
            raise ValueError(
                "upstream_drain_id cannot be empty."
            )

        if not self.downstream_drain_id.strip():
            raise ValueError(
                "downstream_drain_id cannot be empty."
            )

        if (
            self.upstream_drain_id
            == self.downstream_drain_id
        ):
            raise ValueError(
                "A drain cannot connect to itself."
            )

        if not 0.0 < self.flow_fraction <= 1.0:
            raise ValueError(
                "flow_fraction must be greater than 0 "
                "and at most 1."
            )


@dataclass(frozen=True)
class DrainFlowState:
    drain_id: str

    local_catchment_inflow_m3_per_s: float
    upstream_inflow_m3_per_s: float

    total_inflow_m3_per_s: float

    effective_capacity_m3_per_s: float

    conveyed_outflow_m3_per_s: float
    overflow_discharge_m3_per_s: float

    capacity_utilization: float

    data_confidence: float

    risk: DrainRiskAssessment


@dataclass(frozen=True)
class DrainNetworkResult:
    drain_states: tuple[
        DrainFlowState,
        ...
    ]

    total_local_inflow_m3_per_s: float
    total_overflow_m3_per_s: float

    processing_order: tuple[
        str,
        ...
    ]


class DrainNetworkError(
    ValueError
):
    pass


def _build_profile_map(
    profiles: Iterable[
        DrainStaticProfile
    ],
) -> dict[
    str,
    DrainStaticProfile,
]:
    profiles = list(
        profiles
    )

    if not profiles:
        raise DrainNetworkError(
            "At least one drain profile is required."
        )

    profile_map: dict[
        str,
        DrainStaticProfile,
    ] = {}

    for profile in profiles:
        if profile.drain_id in profile_map:
            raise DrainNetworkError(
                f"Duplicate drain_id: {profile.drain_id}"
            )

        profile_map[
            profile.drain_id
        ] = profile

    return profile_map


def _validate_connections(
    connections: Iterable[
        DrainConnection
    ],
    profile_map: dict[
        str,
        DrainStaticProfile,
    ],
) -> list[
    DrainConnection
]:
    connections = list(
        connections
    )

    outgoing_fraction: dict[
        str,
        float,
    ] = {}

    for connection in connections:
        if (
            connection.upstream_drain_id
            not in profile_map
        ):
            raise DrainNetworkError(
                "Unknown upstream drain: "
                f"{connection.upstream_drain_id}"
            )

        if (
            connection.downstream_drain_id
            not in profile_map
        ):
            raise DrainNetworkError(
                "Unknown downstream drain: "
                f"{connection.downstream_drain_id}"
            )

        upstream_id = (
            connection.upstream_drain_id
        )

        outgoing_fraction[
            upstream_id
        ] = (
            outgoing_fraction.get(
                upstream_id,
                0.0,
            )
            + connection.flow_fraction
        )

    for (
        drain_id,
        total_fraction,
    ) in outgoing_fraction.items():
        if total_fraction > 1.0 + 1e-9:
            raise DrainNetworkError(
                "Outgoing flow fractions for "
                f"{drain_id} exceed 1."
            )

    return connections


def _topological_order(
    profile_map: dict[
        str,
        DrainStaticProfile,
    ],
    connections: list[
        DrainConnection
    ],
) -> list[str]:
    """
    Return upstream-to-downstream processing order.

    Cycles are rejected because this simple routing engine assumes
    a directed acyclic drainage representation.

    More advanced hydraulic models may later support loops and
    backwater interactions explicitly.
    """

    indegree = {
        drain_id: 0
        for drain_id in profile_map
    }

    downstream_map: dict[
        str,
        list[str],
    ] = {
        drain_id: []
        for drain_id in profile_map
    }

    for connection in connections:
        indegree[
            connection.downstream_drain_id
        ] += 1

        downstream_map[
            connection.upstream_drain_id
        ].append(
            connection.downstream_drain_id
        )

    queue = sorted(
        drain_id
        for (
            drain_id,
            degree,
        ) in indegree.items()
        if degree == 0
    )

    order: list[str] = []

    while queue:
        current = queue.pop(
            0
        )

        order.append(
            current
        )

        for downstream_id in downstream_map[
            current
        ]:
            indegree[
                downstream_id
            ] -= 1

            if indegree[
                downstream_id
            ] == 0:
                queue.append(
                    downstream_id
                )

                queue.sort()

    if len(order) != len(
        profile_map
    ):
        raise DrainNetworkError(
            "Drainage network contains a cycle."
        )

    return order


def _build_local_inflows(
    *,
    runoff_inputs: Iterable[
        CatchmentRunoffInput
    ],
    profile_map: dict[
        str,
        DrainStaticProfile,
    ],
) -> tuple[
    dict[str, float],
    dict[str, list[float]],
]:
    local_inflow = {
        drain_id: 0.0
        for drain_id in profile_map
    }

    confidence_values: dict[
        str,
        list[float],
    ] = {
        drain_id: []
        for drain_id in profile_map
    }

    for runoff_input in runoff_inputs:
        if (
            runoff_input.drain_id
            not in profile_map
        ):
            raise DrainNetworkError(
                "Catchment runoff references unknown drain: "
                f"{runoff_input.drain_id}"
            )

        result = calculate_catchment_discharge(
            runoff_input
        )

        local_inflow[
            result.drain_id
        ] += (
            result
            .characteristic_discharge_m3_per_s
        )

        confidence_values[
            result.drain_id
        ].append(
            result.data_confidence
        )

    return (
        local_inflow,
        confidence_values,
    )


def _combined_confidence(
    values: list[float],
) -> float:
    if not values:
        return 1.0

    return float(
        sum(values)
        / len(values)
    )


def route_drainage_network(
    *,
    profiles: Iterable[
        DrainStaticProfile
    ],
    connections: Iterable[
        DrainConnection
    ],
    runoff_inputs: Iterable[
        CatchmentRunoffInput
    ],
) -> DrainNetworkResult:
    """
    Route characteristic runoff discharge through a connected
    drainage network.

    For each drain:

        total inflow =
            local catchment inflow
            + conveyed upstream inflow

        conveyed outflow =
            min(total inflow, effective capacity)

        overflow =
            max(
                total inflow - effective capacity,
                0
            )

    Only conveyed flow is propagated downstream.

    This prevents the model from pretending that water exceeding
    the drain's hydraulic capacity remains inside the pipe/channel.

    The resulting overflow discharge will later feed road and
    surface-waterlogging intelligence.

    LIMITATION:

    This is steady/characteristic flow routing, not a dynamic
    hydraulic simulation. It currently does not model:

        travel-time hydrographs
        surcharge
        reverse flow
        storage
        backwater
        inlet restrictions

    Those can later be handled using a higher-fidelity engine such
    as EPA SWMM where data and computation allow.
    """

    profile_map = _build_profile_map(
        profiles
    )

    connections = (
        _validate_connections(
            connections,
            profile_map,
        )
    )

    processing_order = (
        _topological_order(
            profile_map,
            connections,
        )
    )

    (
        local_inflow,
        local_confidence_values,
    ) = _build_local_inflows(
        runoff_inputs=runoff_inputs,
        profile_map=profile_map,
    )

    upstream_inflow = {
        drain_id: 0.0
        for drain_id in profile_map
    }

    upstream_confidence_values: dict[
        str,
        list[float],
    ] = {
        drain_id: []
        for drain_id in profile_map
    }

    outgoing_connections: dict[
        str,
        list[
            DrainConnection
        ],
    ] = {
        drain_id: []
        for drain_id in profile_map
    }

    for connection in connections:
        outgoing_connections[
            connection.upstream_drain_id
        ].append(
            connection
        )

    states: list[
        DrainFlowState
    ] = []

    for drain_id in processing_order:
        profile = profile_map[
            drain_id
        ]

        total_inflow = (
            local_inflow[
                drain_id
            ]
            + upstream_inflow[
                drain_id
            ]
        )

        capacity = (
            calculate_rectangular_manning_capacity(
                profile
            )
        )

        effective_capacity = (
            capacity
            .effective_capacity_m3_per_s
        )

        conveyed_outflow = min(
            total_inflow,
            effective_capacity,
        )

        overflow = max(
            total_inflow
            - effective_capacity,
            0.0,
        )

        confidence_values = (
            local_confidence_values[
                drain_id
            ]
            + upstream_confidence_values[
                drain_id
            ]
        )

        data_confidence = (
            _combined_confidence(
                confidence_values
            )
        )

        dynamic_load = DrainDynamicLoad(
            estimated_inflow_m3_per_s=(
                total_inflow
            ),
            catchment_runoff_mm=0.0,
            rainfall_mm_per_hr=0.0,
            data_confidence=(
                data_confidence
            ),
        )

        risk = assess_drain_overflow_risk(
            profile=profile,
            load=dynamic_load,
        )

        state = DrainFlowState(
            drain_id=drain_id,
            local_catchment_inflow_m3_per_s=float(
                local_inflow[
                    drain_id
                ]
            ),
            upstream_inflow_m3_per_s=float(
                upstream_inflow[
                    drain_id
                ]
            ),
            total_inflow_m3_per_s=float(
                total_inflow
            ),
            effective_capacity_m3_per_s=float(
                effective_capacity
            ),
            conveyed_outflow_m3_per_s=float(
                conveyed_outflow
            ),
            overflow_discharge_m3_per_s=float(
                overflow
            ),
            capacity_utilization=float(
                (
                    total_inflow
                    / effective_capacity
                )
                if effective_capacity > 0.0
                else float("inf")
            ),
            data_confidence=float(
                data_confidence
            ),
            risk=risk,
        )

        states.append(
            state
        )

        for connection in outgoing_connections[
            drain_id
        ]:
            downstream_id = (
                connection
                .downstream_drain_id
            )

            routed_flow = (
                conveyed_outflow
                * connection.flow_fraction
            )

            upstream_inflow[
                downstream_id
            ] += routed_flow

            upstream_confidence_values[
                downstream_id
            ].append(
                data_confidence
            )

    total_local_inflow = sum(
        local_inflow.values()
    )

    total_overflow = sum(
        state.overflow_discharge_m3_per_s
        for state in states
    )

    return DrainNetworkResult(
        drain_states=tuple(
            states
        ),
        total_local_inflow_m3_per_s=float(
            total_local_inflow
        ),
        total_overflow_m3_per_s=float(
            total_overflow
        ),
        processing_order=tuple(
            processing_order
        ),
    )
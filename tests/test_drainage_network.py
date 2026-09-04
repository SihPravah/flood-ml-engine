import pytest

from pravaha_ml.drainage.flow import (
    CatchmentRunoffInput,
)
from pravaha_ml.drainage.models import (
    DrainCapacityProvenance,
    DrainCondition,
    DrainStaticProfile,
)
from pravaha_ml.drainage.network import (
    DrainConnection,
    DrainNetworkError,
    route_drainage_network,
)


def make_drain(
    drain_id: str,
    *,
    width_m: float = 2.0,
    depth_m: float = 1.0,
) -> DrainStaticProfile:
    return DrainStaticProfile(
        drain_id=drain_id,
        width_m=width_m,
        depth_m=depth_m,
        slope_fraction=0.01,
        manning_roughness=0.015,
        blockage_fraction=0.0,
        condition=DrainCondition.GOOD,
        capacity_provenance=(
            DrainCapacityProvenance.VERIFIED
        ),
    )


def test_single_drain_receives_local_runoff():
    result = route_drainage_network(
        profiles=[
            make_drain("D_001")
        ],
        connections=[],
        runoff_inputs=[
            CatchmentRunoffInput(
                catchment_id="C_001",
                drain_id="D_001",
                catchment_area_km2=0.10,
                runoff_mm=5.0,
                response_time_minutes=60.0,
                data_confidence=0.90,
            )
        ],
    )

    assert len(
        result.drain_states
    ) == 1

    state = result.drain_states[
        0
    ]

    assert (
        state.local_catchment_inflow_m3_per_s
        > 0.0
    )

    assert (
        state.upstream_inflow_m3_per_s
        == pytest.approx(0.0)
    )


def test_upstream_flow_reaches_downstream_drain():
    result = route_drainage_network(
        profiles=[
            make_drain("D_A"),
            make_drain("D_B"),
        ],
        connections=[
            DrainConnection(
                upstream_drain_id="D_A",
                downstream_drain_id="D_B",
            )
        ],
        runoff_inputs=[
            CatchmentRunoffInput(
                catchment_id="C_A",
                drain_id="D_A",
                catchment_area_km2=0.05,
                runoff_mm=5.0,
                response_time_minutes=60.0,
                data_confidence=1.0,
            )
        ],
    )

    states = {
        state.drain_id: state
        for state in result.drain_states
    }

    assert (
        states[
            "D_B"
        ].upstream_inflow_m3_per_s
        > 0.0
    )


def test_downstream_combines_local_and_upstream_flow():
    result = route_drainage_network(
        profiles=[
            make_drain("D_A"),
            make_drain("D_B"),
        ],
        connections=[
            DrainConnection(
                upstream_drain_id="D_A",
                downstream_drain_id="D_B",
            )
        ],
        runoff_inputs=[
            CatchmentRunoffInput(
                catchment_id="C_A",
                drain_id="D_A",
                catchment_area_km2=0.05,
                runoff_mm=5.0,
                response_time_minutes=60.0,
                data_confidence=1.0,
            ),
            CatchmentRunoffInput(
                catchment_id="C_B",
                drain_id="D_B",
                catchment_area_km2=0.05,
                runoff_mm=5.0,
                response_time_minutes=60.0,
                data_confidence=1.0,
            ),
        ],
    )

    states = {
        state.drain_id: state
        for state in result.drain_states
    }

    downstream = states[
        "D_B"
    ]

    assert (
        downstream.total_inflow_m3_per_s
        == pytest.approx(
            downstream.local_catchment_inflow_m3_per_s
            + downstream.upstream_inflow_m3_per_s
        )
    )


def test_overflow_is_not_propagated_as_conveyed_flow():
    narrow_upstream = make_drain(
        "D_A",
        width_m=0.20,
        depth_m=0.20,
    )

    downstream = make_drain(
        "D_B"
    )

    result = route_drainage_network(
        profiles=[
            narrow_upstream,
            downstream,
        ],
        connections=[
            DrainConnection(
                upstream_drain_id="D_A",
                downstream_drain_id="D_B",
            )
        ],
        runoff_inputs=[
            CatchmentRunoffInput(
                catchment_id="C_A",
                drain_id="D_A",
                catchment_area_km2=1.0,
                runoff_mm=50.0,
                response_time_minutes=30.0,
                data_confidence=1.0,
            )
        ],
    )

    states = {
        state.drain_id: state
        for state in result.drain_states
    }

    upstream = states[
        "D_A"
    ]

    downstream_state = states[
        "D_B"
    ]

    assert (
        upstream.overflow_discharge_m3_per_s
        > 0.0
    )

    assert (
        downstream_state.upstream_inflow_m3_per_s
        == pytest.approx(
            upstream.conveyed_outflow_m3_per_s
        )
    )

    assert (
        downstream_state.upstream_inflow_m3_per_s
        < upstream.total_inflow_m3_per_s
    )


def test_flow_fraction_splits_discharge():
    result = route_drainage_network(
        profiles=[
            make_drain("D_A"),
            make_drain("D_B"),
            make_drain("D_C"),
        ],
        connections=[
            DrainConnection(
                upstream_drain_id="D_A",
                downstream_drain_id="D_B",
                flow_fraction=0.60,
            ),
            DrainConnection(
                upstream_drain_id="D_A",
                downstream_drain_id="D_C",
                flow_fraction=0.40,
            ),
        ],
        runoff_inputs=[
            CatchmentRunoffInput(
                catchment_id="C_A",
                drain_id="D_A",
                catchment_area_km2=0.05,
                runoff_mm=5.0,
                response_time_minutes=60.0,
                data_confidence=1.0,
            )
        ],
    )

    states = {
        state.drain_id: state
        for state in result.drain_states
    }

    upstream = states[
        "D_A"
    ]

    assert (
        states[
            "D_B"
        ].upstream_inflow_m3_per_s
        == pytest.approx(
            upstream.conveyed_outflow_m3_per_s
            * 0.60
        )
    )

    assert (
        states[
            "D_C"
        ].upstream_inflow_m3_per_s
        == pytest.approx(
            upstream.conveyed_outflow_m3_per_s
            * 0.40
        )
    )


def test_cycle_is_rejected():
    with pytest.raises(
        DrainNetworkError,
        match=(
            "Drainage network contains a cycle"
        ),
    ):
        route_drainage_network(
            profiles=[
                make_drain("D_A"),
                make_drain("D_B"),
            ],
            connections=[
                DrainConnection(
                    upstream_drain_id="D_A",
                    downstream_drain_id="D_B",
                ),
                DrainConnection(
                    upstream_drain_id="D_B",
                    downstream_drain_id="D_A",
                ),
            ],
            runoff_inputs=[],
        )


def test_unknown_downstream_drain_rejected():
    with pytest.raises(
        DrainNetworkError,
        match="Unknown downstream drain",
    ):
        route_drainage_network(
            profiles=[
                make_drain("D_A")
            ],
            connections=[
                DrainConnection(
                    upstream_drain_id="D_A",
                    downstream_drain_id="UNKNOWN",
                )
            ],
            runoff_inputs=[],
        )


def test_outgoing_flow_fraction_above_one_rejected():
    with pytest.raises(
        DrainNetworkError,
        match=(
            "Outgoing flow fractions"
        ),
    ):
        route_drainage_network(
            profiles=[
                make_drain("D_A"),
                make_drain("D_B"),
                make_drain("D_C"),
            ],
            connections=[
                DrainConnection(
                    upstream_drain_id="D_A",
                    downstream_drain_id="D_B",
                    flow_fraction=0.70,
                ),
                DrainConnection(
                    upstream_drain_id="D_A",
                    downstream_drain_id="D_C",
                    flow_fraction=0.60,
                ),
            ],
            runoff_inputs=[],
        )


def test_unknown_runoff_entry_drain_rejected():
    with pytest.raises(
        DrainNetworkError,
        match=(
            "Catchment runoff references unknown drain"
        ),
    ):
        route_drainage_network(
            profiles=[
                make_drain("D_A")
            ],
            connections=[],
            runoff_inputs=[
                CatchmentRunoffInput(
                    catchment_id="C_X",
                    drain_id="UNKNOWN",
                    catchment_area_km2=1.0,
                    runoff_mm=10.0,
                    response_time_minutes=60.0,
                    data_confidence=1.0,
                )
            ],
        )


def test_processing_order_is_upstream_to_downstream():
    result = route_drainage_network(
        profiles=[
            make_drain("D_C"),
            make_drain("D_A"),
            make_drain("D_B"),
        ],
        connections=[
            DrainConnection(
                upstream_drain_id="D_A",
                downstream_drain_id="D_B",
            ),
            DrainConnection(
                upstream_drain_id="D_B",
                downstream_drain_id="D_C",
            ),
        ],
        runoff_inputs=[],
    )

    assert (
        result.processing_order
        == (
            "D_A",
            "D_B",
            "D_C",
        )
    )
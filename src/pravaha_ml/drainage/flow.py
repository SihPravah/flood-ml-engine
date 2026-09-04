from dataclasses import dataclass


@dataclass(frozen=True)
class CatchmentRunoffInput:
    """
    Hydrological runoff entering the urban drainage system from
    one contributing catchment.

    catchment_area_km2:
        Catchment contributing area in square kilometres.

    runoff_mm:
        Effective runoff depth produced by the hydrology engine.

    response_time_minutes:
        Representative catchment response duration.

        PRAVAHA currently uses this as a development approximation
        for converting runoff volume into characteristic discharge.

        This must not be presented as a complete rainfall-runoff
        hydrograph.

    peaking_factor:
        Explicit multiplier representing concentration of runoff
        around peak flow.

        Development default = 1.0.

        This must eventually be calibrated using observed flow/
        waterlogging events or replaced by a proper hydrograph
        model.

    data_confidence:
        Reliability of the input hydrological data.
    """

    catchment_id: str
    drain_id: str

    catchment_area_km2: float
    runoff_mm: float
    response_time_minutes: float

    data_confidence: float

    peaking_factor: float = 1.0

    def __post_init__(self) -> None:
        if not self.catchment_id.strip():
            raise ValueError(
                "catchment_id cannot be empty."
            )

        if not self.drain_id.strip():
            raise ValueError(
                "drain_id cannot be empty."
            )

        if self.catchment_area_km2 <= 0.0:
            raise ValueError(
                "catchment_area_km2 must be greater than 0."
            )

        if self.runoff_mm < 0.0:
            raise ValueError(
                "runoff_mm cannot be negative."
            )

        if self.response_time_minutes <= 0.0:
            raise ValueError(
                "response_time_minutes must be greater than 0."
            )

        if self.peaking_factor <= 0.0:
            raise ValueError(
                "peaking_factor must be greater than 0."
            )

        if not 0.0 <= self.data_confidence <= 1.0:
            raise ValueError(
                "data_confidence must be between 0 and 1."
            )


@dataclass(frozen=True)
class CatchmentDischargeResult:
    catchment_id: str
    drain_id: str

    runoff_volume_m3: float
    characteristic_discharge_m3_per_s: float

    response_time_seconds: float
    peaking_factor: float

    data_confidence: float


def calculate_catchment_discharge(
    runoff: CatchmentRunoffInput,
) -> CatchmentDischargeResult:
    """
    Convert runoff depth over a catchment into a characteristic
    discharge supplied to a drainage entry point.

    Conversion:

        area_m2 =
            catchment_area_km2 * 1,000,000

        runoff_depth_m =
            runoff_mm / 1000

        runoff_volume_m3 =
            runoff_depth_m * area_m2

        characteristic_discharge =
            runoff_volume_m3 / response_time_seconds
            * peaking_factor

    IMPORTANT:

    This is a transparent development approximation.

    It does NOT yet produce a full time-varying hydrograph and
    should not be described as one.
    """

    area_m2 = (
        runoff.catchment_area_km2
        * 1_000_000.0
    )

    runoff_depth_m = (
        runoff.runoff_mm
        / 1000.0
    )

    runoff_volume_m3 = (
        area_m2
        * runoff_depth_m
    )

    response_time_seconds = (
        runoff.response_time_minutes
        * 60.0
    )

    characteristic_discharge = (
        runoff_volume_m3
        / response_time_seconds
        * runoff.peaking_factor
    )

    return CatchmentDischargeResult(
        catchment_id=runoff.catchment_id,
        drain_id=runoff.drain_id,
        runoff_volume_m3=float(
            runoff_volume_m3
        ),
        characteristic_discharge_m3_per_s=float(
            characteristic_discharge
        ),
        response_time_seconds=float(
            response_time_seconds
        ),
        peaking_factor=float(
            runoff.peaking_factor
        ),
        data_confidence=float(
            runoff.data_confidence
        ),
    )
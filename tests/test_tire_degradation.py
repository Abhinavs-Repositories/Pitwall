"""Tests for src/analysis/tire_deg.py"""

import pytest

from src.analysis.tire_deg import (
    CLIFF_THRESHOLD_S,
    MIN_CLEAN_LAPS,
    WARMUP_LAPS,
    CompoundStat,
    _extract_clean_laps,
    _linear_regression,
    _predict_cliff,
    calculate_degradation,
    compare_compound_performance,
    predict_tire_cliff,
)
from src.data.models import DriverState, LapData, Stint, TireCompound


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_laps(
    start: int,
    end: int,
    base_time: float = 95.0,
    slope: float = 0.1,
    pit_in_lap: int | None = None,
    pit_out_lap: int | None = None,
) -> list[LapData]:
    laps = []
    for i, lap_num in enumerate(range(start, end + 1)):
        t = base_time + i * slope
        is_pit_in = lap_num == pit_in_lap
        is_pit_out = lap_num == pit_out_lap
        lap_time = None if (is_pit_in or is_pit_out) else t
        laps.append(
            LapData(
                lap_number=lap_num,
                lap_time=lap_time,
                is_pit_in_lap=is_pit_in,
                is_pit_out_lap=is_pit_out,
            )
        )
    return laps


def _make_stint(lap_start: int, lap_end: int, compound: TireCompound = TireCompound.MEDIUM) -> Stint:
    return Stint(
        stint_number=1,
        compound=compound,
        lap_start=lap_start,
        lap_end=lap_end,
        tyre_age_at_start=0,
    )


def _make_driver(
    driver_number: int,
    laps: list[LapData],
    stints: list[Stint],
    compound: TireCompound = TireCompound.MEDIUM,
) -> DriverState:
    return DriverState(
        driver_number=driver_number,
        name=f"Driver {driver_number}",
        team="Test Team",
        position=1,
        tire_compound=compound,
        lap_times=laps,
        stints=stints,
    )


# ---------------------------------------------------------------------------
# _linear_regression
# ---------------------------------------------------------------------------


def test_linear_regression_perfect_fit():
    import numpy as np

    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([95.0, 95.1, 95.2, 95.3])
    slope, intercept = _linear_regression(x, y)
    assert slope == pytest.approx(0.1, abs=1e-6)
    assert intercept == pytest.approx(95.0, abs=1e-4)


def test_linear_regression_flat():
    import numpy as np

    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([95.0, 95.0, 95.0, 95.0])
    slope, intercept = _linear_regression(x, y)
    assert slope == pytest.approx(0.0, abs=1e-6)


def test_linear_regression_single_point():
    import numpy as np

    x = np.array([0.0])
    y = np.array([95.0])
    slope, intercept = _linear_regression(x, y)
    assert slope == 0.0


# ---------------------------------------------------------------------------
# _predict_cliff
# ---------------------------------------------------------------------------


def test_predict_cliff_returns_none_for_negative_slope():
    cliff = _predict_cliff(slope=-0.05, intercept=95.0, stint_best=95.0, lap_start=1)
    assert cliff is None


def test_predict_cliff_detects_cliff():
    # With slope 0.1 s/lap, best=95: cliff at relative lap 25 (95+25*0.1=97.5 → 95+2.5=cliff)
    cliff = _predict_cliff(slope=0.1, intercept=95.0, stint_best=95.0, lap_start=1)
    assert cliff is not None
    # At relative lap 25: 95 + 0.1*25 = 97.5 = 95 + 2.5 ✓
    assert cliff == 26  # lap_start=1, so cliff at relative 25 → lap 26


def test_predict_cliff_returns_none_if_beyond_lookahead():
    # Very slow degradation — cliff never reached in 60 laps
    cliff = _predict_cliff(slope=0.001, intercept=95.0, stint_best=95.0, lap_start=1, max_look_ahead=60)
    assert cliff is None


# ---------------------------------------------------------------------------
# _extract_clean_laps
# ---------------------------------------------------------------------------


def test_extract_clean_laps_filters_warmup():
    stint = _make_stint(lap_start=1, lap_end=30)
    laps = _make_laps(1, 30, base_time=95.0, slope=0.05)
    clean = _extract_clean_laps(laps, stint, sc_laps=set())
    # Warmup laps 1 and 2 should be excluded
    assert all(l.lap_number >= 1 + WARMUP_LAPS for l in clean)


def test_extract_clean_laps_filters_pit_laps():
    stint = _make_stint(lap_start=1, lap_end=27)
    laps = _make_laps(1, 27, base_time=95.0, slope=0.05, pit_in_lap=27)
    clean = _extract_clean_laps(laps, stint, sc_laps=set())
    assert all(not l.is_pit_in_lap for l in clean)


def test_extract_clean_laps_filters_safety_car():
    stint = _make_stint(lap_start=1, lap_end=30)
    laps = _make_laps(1, 30, base_time=95.0, slope=0.05)
    clean = _extract_clean_laps(laps, stint, sc_laps={10, 11, 12})
    assert all(l.lap_number not in {10, 11, 12} for l in clean)


def test_extract_clean_laps_removes_outliers():
    stint = _make_stint(lap_start=1, lap_end=20)
    laps = _make_laps(1, 20, base_time=95.0, slope=0.0)
    # Inject a massive outlier
    laps[10] = LapData(lap_number=11, lap_time=120.0)
    clean = _extract_clean_laps(laps, stint, sc_laps=set())
    assert all((l.lap_time or 0) < 115.0 for l in clean)


# ---------------------------------------------------------------------------
# calculate_degradation
# ---------------------------------------------------------------------------


def test_calculate_degradation_returns_none_for_too_few_laps():
    stint = _make_stint(lap_start=1, lap_end=5)  # only 5 laps total; after warmup + filter: < MIN
    laps = _make_laps(1, 5, base_time=95.0, slope=0.1)
    result = calculate_degradation(laps, stint)
    assert result is None


def test_calculate_degradation_basic():
    stint = _make_stint(lap_start=1, lap_end=30, compound=TireCompound.MEDIUM)
    laps = _make_laps(1, 30, base_time=95.0, slope=0.1)
    result = calculate_degradation(laps, stint)
    assert result is not None
    assert result.compound == TireCompound.MEDIUM
    assert result.deg_rate_per_lap == pytest.approx(0.1, abs=0.01)


def test_calculate_degradation_cliff_predicted():
    # slope=0.1 → cliff in ~25 laps from stint start
    stint = _make_stint(lap_start=1, lap_end=30)
    laps = _make_laps(1, 30, base_time=95.0, slope=0.1)
    result = calculate_degradation(laps, stint)
    assert result is not None
    assert result.predicted_cliff_lap is not None
    assert result.predicted_cliff_lap > 1


def test_calculate_degradation_flat_no_cliff():
    # Zero degradation → no cliff
    stint = _make_stint(lap_start=1, lap_end=30)
    laps = _make_laps(1, 30, base_time=95.0, slope=0.0)
    result = calculate_degradation(laps, stint)
    assert result is not None
    assert result.predicted_cliff_lap is None


def test_calculate_degradation_with_safety_car_laps():
    stint = _make_stint(lap_start=1, lap_end=30)
    laps = _make_laps(1, 30, base_time=95.0, slope=0.1)
    result = calculate_degradation(laps, stint, safety_car_laps={10, 11})
    assert result is not None  # should still work with fewer clean laps


# ---------------------------------------------------------------------------
# predict_tire_cliff
# ---------------------------------------------------------------------------


def test_predict_tire_cliff_delegates_to_calculate_degradation():
    stint = _make_stint(lap_start=1, lap_end=30)
    laps = _make_laps(1, 30, base_time=95.0, slope=0.12)
    cliff = predict_tire_cliff(laps, stint)
    deg = calculate_degradation(laps, stint)
    assert cliff == (deg.predicted_cliff_lap if deg else None)


# ---------------------------------------------------------------------------
# compare_compound_performance
# ---------------------------------------------------------------------------


def test_compare_compound_performance_sorted_by_pace():
    laps_fast = _make_laps(1, 30, base_time=93.0, slope=0.05)
    laps_slow = _make_laps(1, 30, base_time=96.0, slope=0.05)
    stint = _make_stint(1, 30, TireCompound.HARD)

    driver_fast = _make_driver(1, laps_fast, [stint], TireCompound.HARD)
    driver_slow = _make_driver(2, laps_slow, [stint], TireCompound.HARD)

    stats = compare_compound_performance([driver_fast, driver_slow], TireCompound.HARD)

    assert len(stats) == 2
    # Fastest first
    assert stats[0].driver_number == 1
    assert stats[0].median_pace < stats[1].median_pace


def test_compare_compound_performance_skips_wrong_compound():
    laps = _make_laps(1, 30, base_time=95.0, slope=0.05)
    stint_medium = _make_stint(1, 30, TireCompound.MEDIUM)
    driver = _make_driver(1, laps, [stint_medium], TireCompound.MEDIUM)

    # Query SOFT — driver is on MEDIUM, should return empty
    stats = compare_compound_performance([driver], TireCompound.SOFT)
    assert stats == []


def test_compare_compound_performance_skips_insufficient_laps():
    laps = _make_laps(1, 4, base_time=95.0, slope=0.1)  # too few clean laps
    stint = _make_stint(1, 4, TireCompound.SOFT)
    driver = _make_driver(1, laps, [stint], TireCompound.SOFT)

    stats = compare_compound_performance([driver], TireCompound.SOFT)
    assert stats == []

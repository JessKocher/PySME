# -*- coding: utf-8 -*-
"""Tests for the RBF-based atmosphere-grid interpolation path (RbfGrid/interpolate_RBF)."""
import numpy as np

from pysme.abund import Abund
from pysme.atmosphere.atmosphere import AtmosphereGrid
from pysme.atmosphere.interpolation import AtmosphereInterpolator


def _make_synthetic_grid(
    source, teffs, loggs, monhs, temp_value, ndep=3, geom="PP",
    opflag_value=1, wlstd_value=5000.0, abund_value=0.0,
):
    """
    Build a tiny in-memory AtmosphereGrid spanning the Cartesian product of
    teffs x loggs x monhs, with every depth point of ``temp`` set to
    ``temp_value`` and every other field set to a fixed, physically-harmless
    placeholder - enough for RbfGrid to build an interpolator, using the same
    field-assignment convention SavFile uses when loading a real grid.
    """
    combos = [(t, g, m) for t in teffs for g in loggs for m in monhs]
    natmo = len(combos)
    grid = AtmosphereGrid(natmo, ndep)
    grid.source = source
    grid.geom = geom
    grid["teff"] = [c[0] for c in combos]
    grid["logg"] = [c[1] for c in combos]
    grid["monh"] = [c[2] for c in combos]
    grid["vturb"] = 0.0
    grid["lonh"] = 0.0
    grid["wlstd"] = wlstd_value
    grid["radius"][:] = 1 if geom == "PP" else 1e11
    grid["temp"] = temp_value
    grid["tau"] = np.tile(np.linspace(1e-4, 1e-1, ndep), (natmo, 1))
    grid["rhox"] = np.tile(np.linspace(1e-3, 1.0, ndep), (natmo, 1))
    grid["rho"] = 1.0
    grid["xna"] = 1.0
    grid["xne"] = 1.0
    grid["opflag"] = opflag_value
    grid["abund"] = abund_value
    if geom == "SPH":
        grid["height"] = np.tile(np.linspace(0.0, 1.0, ndep), (natmo, 1))
    return grid


def test_rbf_cache_invalidated_on_source_change():
    teffs, loggs, monhs = [4900.0, 5100.0], [3.8, 4.2], [-0.2, 0.2]
    grid_a = _make_synthetic_grid("grid_a", teffs, loggs, monhs, temp_value=5000.0)
    grid_b = _make_synthetic_grid("grid_b", teffs, loggs, monhs, temp_value=6000.0)

    interpolator = AtmosphereInterpolator(interp="RBF")
    atmo_a = interpolator.interp_atmo_grid(grid_a, 5000.0, 4.0, 0.0)
    # Same interpolator instance, different grid: must not reuse grid_a's cached RbfGrid.
    atmo_b = interpolator.interp_atmo_grid(grid_b, 5000.0, 4.0, 0.0)

    fresh_interpolator = AtmosphereInterpolator(interp="RBF")
    atmo_b_fresh = fresh_interpolator.interp_atmo_grid(grid_b, 5000.0, 4.0, 0.0)

    assert np.allclose(atmo_a.temp, 5000.0)
    assert np.allclose(atmo_b.temp, atmo_b_fresh.temp)
    assert not np.allclose(atmo_b.temp, atmo_a.temp)


def test_rbf_handles_single_valued_metallicity_axis():
    # A single metallicity value (like the real spherical MARCS grid) must
    # not crash the per-axis step-size computation in initialize_gridpoints.
    grid = _make_synthetic_grid(
        "single_monh_grid", teffs=[4900.0, 5100.0], loggs=[3.8, 4.2], monhs=[0.0],
        temp_value=5000.0,
    )
    interpolator = AtmosphereInterpolator(interp="RBF")
    atmo = interpolator.interp_atmo_grid(grid, 5000.0, 4.0, 0.0)
    assert np.all(np.isfinite(atmo.temp))


def test_rbf_preserves_opflag_wlstd_and_abundance():
    # opflag, wlstd, and the abundance pattern all deliberately differ here
    # from Atmo()'s hardcoded defaults ([1]*20, 5000.0, an "empty" pattern).
    teffs, loggs, monhs = [4900.0, 5100.0], [3.8, 4.2], [-0.2, 0.2]
    grid = _make_synthetic_grid(
        "metadata_grid", teffs, loggs, monhs, temp_value=5000.0,
        opflag_value=0, wlstd_value=6000.0, abund_value=-5.0,
    )
    interpolator = AtmosphereInterpolator(interp="RBF")
    atmo = interpolator.interp_atmo_grid(grid, 5000.0, 4.0, 0.0)

    assert np.array_equal(atmo.opflag, np.zeros(20, dtype=atmo.opflag.dtype))
    assert atmo.wlstd == 6000.0
    expected_abund = Abund(monh=atmo.monh, pattern=np.full(99, -5.0), type="sme")
    assert atmo.abund.get_pattern_abundance("Fe") == expected_abund.get_pattern_abundance("Fe")

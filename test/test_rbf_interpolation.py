# -*- coding: utf-8 -*-
"""Tests for the RBF-based atmosphere-grid interpolation path (RbfGrid/interpolate_RBF)."""
import numpy as np

from pysme.abund import Abund, elements_dict
from pysme.atmosphere.atmosphere import AtmosphereGrid
from pysme.atmosphere.interpolation import AtmosphereInterpolator

# Approximate solar photospheric abundances (H=12 scale), in the standard
# element order (H, He, Li, Be, B, C, N, O, ...). Illustrative values in the
# same ballpark as Grevesse & Sauval (1998)/Asplund et al. (2009) - not a
# citation-grade table, just realistic enough that test fixtures don't look
# like an arbitrary constant. NaN marks elements with no well-established
# solar value (unstable isotopes Tc/Pm, or heavy elements beyond Bi besides
# Th/U), matching how a real solar pattern leaves them untracked.
_SOLAR_H12 = np.array([
    12.00, 10.93,                                                  # H, He
    1.05, 1.38, 2.70, 8.43, 7.83, 8.69, 4.56, 7.93,                 # Li-Ne
    6.24, 7.60, 6.45, 7.51, 5.41, 7.12, 5.50, 6.40,                 # Na-Ar
    5.03, 6.34, 3.15, 4.95, 3.93, 5.64, 5.43, 7.50,                 # K-Fe
    4.99, 6.22, 4.19, 4.56, 3.04, 3.65, 2.30, 3.34,                 # Co-Se
    2.54, 3.25, 2.52, 2.87, 2.21, 2.58, 1.46, 1.88,                 # Br-Mo
    np.nan, 1.75, 0.91, 1.57, 0.94, 1.71, 0.80, 2.04,               # Tc-Sn
    1.01, 2.18, 1.55, 2.24, 1.08, 2.18, 1.10, 1.58,                 # Sb-Ce
    0.72, 1.42, np.nan, 0.96, 0.52, 1.07, 0.30, 1.10,               # Pr-Dy
    0.48, 0.92, 0.10, 0.84, 0.10, 0.85, -0.12, 0.85,                # Ho-W
    0.26, 1.40, 1.38, 1.62, 0.92, 1.17, 0.90, 1.75,                 # Re-Pb
    0.65, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.02,     # Bi-Th
    np.nan, -0.54, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,  # Pa-Cf
    np.nan,                                                         # Es
])
assert len(_SOLAR_H12) == 99


def _make_synthetic_grid(
    source, teffs, loggs, monhs, temp_value, ndep=3, geom="PP",
    opflag_value=1, wlstd_value=5000.0, abund_shift=0.0,
):
    """
    Build a tiny in-memory AtmosphereGrid spanning the Cartesian product of
    teffs x loggs x monhs, with every depth point of ``temp`` set to
    ``temp_value`` and every other field set to a fixed, physically-harmless
    placeholder - enough for RbfGrid to build an interpolator, using the same
    field-assignment convention SavFile uses when loading a real grid.

    The abundance pattern is realistic solar (``_SOLAR_H12``) with
    ``abund_shift`` added to the metals (index 2+) - a constant, or, for
    tests that need abundance to vary with metallicity, a callable of monh
    (mimicking a realized, metallicity-baked-in pattern: solar + monh).
    Stored in "sme" format (via ``Abund.totype``) like a real grid, so
    reading it back through ``_realized_abund_h12`` round-trips correctly.
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
    for i, (_, _, m) in enumerate(combos):
        shift = abund_shift(m) if callable(abund_shift) else abund_shift
        intended_h12 = _SOLAR_H12.copy()
        intended_h12[2:] += shift
        grid["abund"][i] = Abund.totype(intended_h12, "sme", raw=True)
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


def test_rbf_preserves_opflag_and_wlstd():
    # opflag and wlstd deliberately differ here from Atmo()'s hardcoded
    # defaults ([1]*20, 5000.0).
    teffs, loggs, monhs = [4900.0, 5100.0], [3.8, 4.2], [-0.2, 0.2]
    grid = _make_synthetic_grid(
        "metadata_grid", teffs, loggs, monhs, temp_value=5000.0,
        opflag_value=0, wlstd_value=6000.0,
    )
    interpolator = AtmosphereInterpolator(interp="RBF")
    atmo = interpolator.interp_atmo_grid(grid, 5000.0, 4.0, 0.0)

    assert np.array_equal(atmo.opflag, np.zeros(20, dtype=atmo.opflag.dtype))
    assert atmo.wlstd == 6000.0


def test_rbf_abundance_stays_consistent_with_interpolated_metallicity():
    # Every grid point's abundance is solar shifted by exactly its own monh
    # (a stand-in for a realized, metallicity-baked-in pattern, as confirmed
    # against the real marcs2014.sav grid). teff/logg are irrelevant to the
    # abundance here, only monh matters.
    teffs, loggs, monhs = [4900.0, 5100.0], [3.8, 4.2], [-2.0, -1.0, 0.0]
    grid = _make_synthetic_grid(
        "abundance_grid", teffs, loggs, monhs, temp_value=5000.0,
        abund_shift=lambda monh: monh,
    )
    interpolator = AtmosphereInterpolator(interp="RBF")

    # Off-grid metallicity, roughly midway between two grid values.
    query_monh = -1.4
    atmo = interpolator.interp_atmo_grid(grid, 5000.0, 4.0, query_monh)

    # atmo.monh itself is only approximately the query (RBF is a fit, not
    # exact interpolation at off-grid points), so compare the abundance
    # against a pattern built the same way at atmo's *own* interpolated
    # monh - this is the actual bug fix under test: the returned abundance
    # must reflect the atmosphere's own reported metallicity, not whichever
    # grid point happened to be nearest.
    expected_fe = _SOLAR_H12[elements_dict["Fe"]] + atmo.monh

    assert np.isclose(atmo.abund.get_pattern_abundance("Fe"), expected_fe, atol=1e-6)

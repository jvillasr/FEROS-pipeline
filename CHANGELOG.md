# Changelog

## Unreleased

- Restore the required G2, K5 and M2 velocity masks omitted from the initial
  source snapshot, and check all 31 required data files during environment
  validation.

This FEROS-specific MPIA adaptation derives from CERES by Rafael Brahm,
Andrés Jordán and Néstor Espinoza. See README.md and License.md for attribution
and source provenance.

- Add `feros-reduce` for planning and running FEROS reductions with suggested
  worker settings and isolated output directories for each target.
- Build calibration products once, verify compatibility and reuse them through
  an explicit calibration directory before collecting science products.
- Correct result-table timestamp writing to use the current spectrum's value.
- Add FEROS diagnostics for reduction comparisons, wavelength coverage, spectral
  artefacts, calibration compatibility and timing.
- Provide pinned Python 3.9 dependencies and source build support for the Marsh
  and CCF extensions.
- Publish the FEROS source, required reference tables, diagnostics and build
  support. Personal launchers, legacy correction tools, unused libraries,
  compiled artefacts and private development records are retained locally.

The scientific extraction, wavelength grid, blaze correction and normalisation
choices are preserved. Target-level parallel processing and calibration reuse
remain optional; the recorded validation limits are in docs/validation.md.

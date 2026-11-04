# FEROS scope and validation

## Distributed files

| Files | Purpose |
| --- | --- |
| `feros-reduce`, `scripts/reduce_feros_night.py`, `scripts/reduce_targets_parallel.py` | FEROS planning, calibration reuse, isolated target workers and product collection. |
| `feros/ferospipe.py`, `feros/ferosutils.py`, `feros/black_holes.py` | The reduction engine, detector handling and Astropy timing/velocity corrections. |
| `feros/wavcals/` | Wavelength-reference line lists read by the FEROS calibration code. Their inherited filenames contain `harps`; they are required by this FEROS implementation. |
| `data/xc_masks/` | G2, K5 and M2 masks for radial-velocity measurements. |
| `utils/GLOBALutils/`, `utils/Correlation/` | Shared CERES routines imported by FEROS, with the two reference tables used by the classification routine. |
| `utils/OptExtract/Marsh.c`, `utils/OptExtract/setup.py`, `utils/CCF/CCF.f`, `utils/CCF/CCF.pyf` | Sources and build metadata for the two native extension modules. |
| Remaining `scripts/` tools | Environment checks, extension builds and FEROS reduction diagnostics. |
| Project metadata, README, changelog and licence | Reproducible dependencies, usage, attribution and release scope. |

The publication tree contains no compiled binaries, observations, reduction
products, personal launchers, legacy correction tools or development logbooks.
The original CERES MIT notice is preserved. The local development history and
archives are separate from the parentless publication snapshot.

## Supported workflow

The supported FEROS workflow uses the existing Astropy timing-correction path.
The unused SSEphem/SOFA bundle and its inherited alternative backend are not
part of this distribution. Shared CERES modules retain their upstream utility
functions; this release does not claim support for other instruments or unused
legacy options.

The optional `-do_class` step needs the external Coelho model grid. It is not
included in the repository. Native Marsh and CCF extensions must be built for
the active Python version and platform before running a reduction.

## Existing scientific evidence

The recorded full-night validation used the 27 June 2026 FEROS dataset with
10 targets and 11 science exposures. Target-parallel processing produced the
same product filenames and result-row count as the serial baseline. Of the
compared numeric result fields, 144 matched and 10 timestamp values differed
as expected because the serial baseline predated the per-spectrum timestamp
fix. These comparisons do not establish performance for every observing night.

The inherited merged wavelength grid remains 3900–6770 Å. Extraction, blaze
correction and normalisation defaults are unchanged. Full-range wavelength and
artefact diagnostics remain experimental.

## Fresh installation verification

On 4 November 2026, GitHub checkout `c980298` was built on Linux with Python
3.9.25, uv 0.13.7, GCC/GFortran and GSL 2.6. Runtime and reduction checks then
used `c56766a`, which adds worker-cache handling without changing scientific
code or dependencies. The test used a new environment,
newly built Marsh/CCF extensions, isolated caches and an empty Astropy
configuration directory. No development-checkout source, binaries or environment
were used by that installation.

The documented dependency/build commands succeeded. All 18 checked imports came
from the fresh checkout or its new environment. All 31 required reference files
were present, and deliberately hiding one velocity mask made the environment
checker fail. Small compiled extraction and correlation calls, Astropy timing
corrections, command help and read-only FEROS-night planning passed.

The initial source release omitted the three velocity masks; the fresh test
caught the omission. The original masks have been restored unchanged and the
environment checker now verifies required runtime data as well as imports.

The fresh checkout also completed a one-target reduction from the 27 June 2026
raw night in 44 minutes 49 seconds, building calibrations from raw. The
compatibility check accepted all 304 required calibration products. The target
worker reused only those new products and collected one spectrum and one result
row. All 15 spectrum data arrays were finite, and the merged wavelength grid was
strictly increasing.

The six main spectrum arrays matched exactly between the calibration-seed
extraction and the target worker. Against the previous validated Python 3.9
reduction, all 14 numeric result-table fields matched at their written precision,
including velocity and timestamp. Extracted and blaze-corrected flux matched
exactly; the largest normalised merged-flux difference was about `7.1e-11`.
The original merged grid and scientific defaults are unchanged.

This verifies a fresh Linux installation and one science exposure, including
calibration generation and reuse. It does not test other observing nights,
concurrent target workers in this installation, other platforms or the optional
external classification grid.

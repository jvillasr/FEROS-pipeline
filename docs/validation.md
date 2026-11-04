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

## Publication checks and limits

The release audit traces local Python imports and required reference files,
checks that archived files are preserved, excludes private and generated files
from the publication tree and history, and verifies the retained scientific
sources against the pre-cleanup snapshot. Import and planning smoke checks use
the existing development environment. No fresh extension build or new full-night
reduction is part of this packaging check.

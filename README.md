# FEROS-pipeline

A FEROS-specific adaptation of [CERES](https://github.com/rabrahm/ceres) for
reduction workflows at the Max Planck Institute for Astronomy (MPIA).
Maintained by [Jaime Villaseñor](https://github.com/jvillasr).

This repository develops the FEROS reduction workflow and the supporting CERES
utilities it needs. Its focus is reliable reductions, isolated target-level
parallel processing, compatible calibration reuse, and diagnostics for spectral
coverage and artefacts. Development takes place separately from the shared MPIA
CERES installation.

## Credit and provenance

CERES was developed by **Rafael Brahm, Andrés Jordán and Néstor Espinoza**.
The original pipeline, algorithms and supporting routines come from their
[CERES repository](https://github.com/rabrahm/ceres). Please cite
[Brahm, Jordán & Espinoza (2017), PASP, 129, 034002](https://doi.org/10.1088/1538-3873/aa5455)
when using this pipeline in research.

This is a downstream FEROS adaptation with its own development history. The
initial local commit, `5a90fe2`, copied a FEROS subset of MPIA's shared CERES
working tree, whose recorded Git commit was
`813ad55f5a3ae39cf64e272adc68fe1a5f041f20`. That working tree contained local
changes, so this baseline is not an unmodified checkout of the original CERES
repository. This repository does not imply endorsement by the CERES authors.

The original CERES MIT copyright and permission notice is retained in
[License.md](License.md). Source notices within the retained CERES utilities are
preserved.

## Environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python
3.9. Dependencies are pinned in `pyproject.toml` and `uv.lock`:

```bash
uv sync --locked
uv run python scripts/build_extensions.py
uv run python scripts/check_environment.py
```

This source distribution requires C and Fortran compilers, matching Python
development headers and the GSL development libraries for the Marsh and CCF
extensions. Building the pinned PycURL dependency from source also needs libcurl
development files and `curl-config` on the command path; see the
[PycURL installation instructions](https://github.com/pycurl/pycurl/blob/REL_7_45_1/INSTALL.rst).
`uv sync` alone does not build the CERES modules; use the build helper shown
above. Compiled binaries are kept local and must be rebuilt for the active
Python version and platform.
The helper places build work under `FEROS_BUILD_ROOT` when set, then `TMPDIR`,
or otherwise the normal user cache. Build work is retained for inspection;
the printed directory can be removed after it is no longer needed.
Validation includes a fresh Linux installation and a reduction from raw on the
recorded validation night; installation on other platforms remains to be tested.

## Run a FEROS reduction

From the repository root:

```bash
./feros-reduce /path/to/raw/4k_directory --dry-run
./feros-reduce /path/to/raw/4k_directory
```

The first command displays the plan. The second reduces all discovered FEROS
science targets and writes products under `outputs/<raw-dir-name>_red/` by
default. It builds calibrations once, checks their compatibility and reuses them
through `-calibdir`. Target workers have isolated output directories, followed
by an explicit collection step.

The runner suggests worker settings from CPU count and current load. In an
interactive shell, prompts let you adjust calibration pools, target workers,
target pools and the output directory. To accept the suggestions without prompts:

```bash
./feros-reduce /path/to/raw/4k_directory --yes
```

Use a named `tmux` session for long reductions. Inspect `./feros-reduce --help`
for the available options. The lower-level interfaces remain
`scripts/reduce_targets_parallel.py` and `feros/ferospipe.py`.

## Validation and outputs

Target-level parallel reduction and calibration reuse are **optional** workflows,
validated on the recorded 2026-06-27 night. Full-range wavelength and artefact
diagnostics remain **experimental**. See [scope and validation](docs/validation.md)
for the retained dependencies, evidence and limitations. The supported workflow
uses the existing Astropy timing-correction path. Optional spectral
classification requires an external Coelho model grid.

Final collected products include extracted spectra, radial-velocity results
and optional cross-correlation plots. The inherited merged spectrum uses a
3900–6770 Å grid; per-order spectra can cover a wider range. This adaptation has
not changed the scientific wavelength grid, blaze correction or normalisation
defaults.

Raw observations, reduction products, local environments and caches should stay
outside version control. Generated contents of `outputs/` are ignored. Private
development instructions and logbooks, personal launchers, legacy correction
tools and unused libraries are excluded from the publication branch. The full
development history is retained locally.

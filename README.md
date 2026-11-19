# Fault-converter datasets and experiment archive

Data and archived runs for the [fault-converter experiment code](https://github.com/igmsalinas/fault-converters-code) and the [variable-load FRA manuscript](https://github.com/igmsalinas/variable-load-fra-paper). Repository access requires GitHub permissions. The publication snapshot used in the variable-load reproducibility review is [4950ecf](https://github.com/igmsalinas/fault-converters-data/tree/4950ecf604fab8b63395201b2d4a28af88fae8e8), paired with [code snapshot 9e13ccd](https://github.com/igmsalinas/fault-converters-code/tree/9e13ccdd263346e2eadd15102fe096393a64d38a).

## Variable-load benchmark

[data/buck/buck_data_v2/dataset_02](data/buck/buck_data_v2/dataset_02) contains the simulation parameters, manifest and 800 raw sweeps in ordinary Git. Each sweep has 101 logarithmically spaced frequencies from 100 Hz to 50 kHz, with control-to-output magnitude in decibels and phase in degrees. The sampled load range is 2–8 Ω. The manifest contains filenames, labels, component multipliers relative to nominal, and the number of degraded components.

| Partition | Healthy | Faulty |
| --- | ---: | ---: |
| Training | 300 | 0 |
| Fit-validation | 100 | 90 |
| Test | 100 | 210 |
| Full archive | 500 | 300 |

There are 161 single-component and 139 compound faults in the full dataset. The healthy-only validation subset consists of the 100 healthy fit-validation rows. In the test partition, 113 faulty sweeps have one degraded component and 97 have multiple degraded components. The split indices refer to the loader's sorted sweep order. Use the accompanying manifest and pinned loader when joining them to file identities. The Dataset 02 copy in the audited code snapshot matches these files after normalizing line endings.

The [variable-load experiment archive](experiments/dataset_02_mlwa_study) contains the saved split, preprocessing state, checkpoints and intermediate evaluation artifacts. The paper's final ten-seed results are in the [code repository's publication records](https://github.com/igmsalinas/fault-converters-code/tree/9e13ccdd263346e2eadd15102fe096393a64d38a/papers/results/mlwa_study_dataset02).

## Fixed-load comparison in the manuscript

The [corrected Dataset 01 campaign](experiments/dataset_01_run_06_corrected) contains saved predictions and evaluation provenance for the comparison. Its full test cohort consists of 520 healthy and 748 faulty states. Excluding the 82 cases with only low-side switch degradation leaves 520 healthy and 666 faulty states. The simulation-results ZIP under `data/buck` contains a separate legacy sweep grid. The exact raw Dataset 01 cohort and its filename manifest remain to be archived.

## Download the needed files

The repository uses [Git LFS](https://git-lfs.com/) for large binaries. Prefer a pointer-only checkout and hydrate selected paths. The audited snapshot references roughly 30 GB of unique LFS objects.

~~~sh
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/igmsalinas/fault-converters-data.git
cd fault-converters-data
git lfs install --local
git lfs pull --include='experiments/dataset_02_mlwa_study/split_indices.npz'
~~~

For checkpoint replay, expand the inclusion filter to the exact checkpoint directory and include all weight components in it. Ensure the requested files have been materialized before passing them to NumPy or a model loader. For a release or review package, verify retrieval from a clean checkout at the cited revision.

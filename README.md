# hazel-dormouse-classifier
A machine learning classifier designed to detect dormouse calls in bioacoustics data

# Hazel Dormouse Classifier: End-to-End Setup and Run Guide  #TODO: Update user guide description

This guide explains how to set up and run the hazel dormouse classifier on Polar from a fresh user account.

~~The pipeline reads ultrasonic recordings from the JASMIN object store, stages audio files temporarily, runs BatDetect2 inference, writes chunk-level outputs, and then combines results across chunks.~~

---

# 1. Folder structure

#TODO: Update section

```text
batdetect_ultrasonic/
├── README.md
├── configs/
├── docs/
├── logs/
├── outputs/
├── scripts/
│   ├── generate_ultrasonic_keys.py
│   ├── run_batdetect_chunk.py
│   └── combine_batdetect_outputs.py
├── slurm/
│   └── run_batdetect_array.slurm
├── tests/
└── tmp/
```

## Key folders

### `configs/`

Stores reusable configuration files for countries, deployments, runtime settings, chunk sizes, and future BatDetect runtime parameters.

### `docs/`

Contains workflow documentation, technical notes, troubleshooting guides, operational procedures, QA standards, and future processing documentation.

### `logs/`

Contains Slurm `.out` and `.err` files generated during BatDetect processing.

### `outputs/`

Contains:

* generated key manifests
* chunk JSON files
* BatDetect inference outputs
* summary JSON files
* combined deployment outputs
* no-detection logs

### `scripts/`

Contains CEH wrapper scripts used for:

* object-store interaction
* ultrasonic key generation
* BatDetect execution
* structured CSV extraction
* deployment-level output combination
* metadata handling
* provenance tracking

### `slurm/`

Contains reusable Slurm submission scripts for scalable processing on JASMIN.

### `tests/`

Contains future validation scripts, testing notebooks, diagnostic workflows, runtime benchmarking, and QA procedures.

### `tmp/`

Used for temporary staged ultrasonic recordings downloaded from the object store.

Temporary staged files are deleted automatically after each chunk unless:

```bash
--keep-temp
```

is used for debugging.

---

# 2. Install Miniforge (if Conda is not available)

First check whether Conda is already available:

```bash
conda --version
```

If this returns:

```text
conda: command not found
```

install Miniforge into your home directory.

Move to your home directory:

```bash
cd ${HOME}
```

Download Miniforge:

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
```

Run the installer:

```bash
bash Miniforge3-Linux-x86_64.sh
```

Accept the licence and install to the default location:

```text
${HOME}/miniforge3
```

Then load Conda:

```bash
source ${HOME}/miniforge3/etc/profile.d/conda.sh
```

Check installation:

```bash
conda --version
```

---

# 3. Get the repository

Clone the repository into your JASMIN home space.

```bash
cd ${HOME}

git clone <REPOSITORY_URL> hazel-dormouse-classifier
```

Move into the folder:

```bash
cd ${HOME}/hazel-dormouse-classifier
```

Expected structure:  #TODO: Update expected structure

```text
batdetect_ultrasonic/
├── README.md
├── configs/
├── docs/
├── logs/
├── outputs/
├── scripts/
├── slurm/
├── tests/
└── tmp/
```

---

# 4. Create and maintain the Conda environment

Load Conda:

```bash
source ${HOME}/miniforge3/etc/profile.d/conda.sh
```

## Environment files in this repository

### Manual environment file: For basic setup

The repository includes a hand-maintained Conda environment definition for the dormouse classifier under:

```text
environments/dormouse-classifier.yml
```

The hand-maintained environment file records the packages intentionally required by the project. Some of these packages (e.g. `opensoundscape`) are installed via pip rather than conda, so this file is maintained manually rather than generated using `conda env export --from-history`.

Maintaining this manually helps keep the environment tidy and avoids installing unnecessary packages that might be left over from experimental code.

### Exported runtime environment files: For reproducing previous results

There may also be a set of exported environment files which record the complete software environment used to generate a particular model or output.

---

## Create the dormouse classifier environment

```bash
conda env create -f environments/dormouse-classifier.yml
```

Activate:

```bash
conda activate dormouse-classifier
```

Check opensoundscape imports successfully:

```bash
python -c "import opensoundscape; print('opensoundscape OK')"
```
---

## Documenting the environment used for a model run

As well as the packages explicitly installed by `environments/dormouse-classifier.yml`, your environment will contain other packages that might affect your results or whether the code runs without errors.

It's worth documenting the state of the full environment in which any important outputs, such as trained models or classifier outputs, were generated. This makes it easier for others (or future you) to reproduce your results.

After producing any outputs that you want to document, generate an exported YAML:

```bash
conda env export -n dormouse-classifier > environments/dormouse-classifier-<output-name>.yml
```

Replace `<output-name>` with a label to identify which output the environment file is associated with.

Commit the updated YAML file to Git so the repository contains the information needed to reproduce your work.

---
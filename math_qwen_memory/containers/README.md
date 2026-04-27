# Private LUMI PyTorch container with extra pip packages

`build_qwen36.sh` produces a project-local copy of LUMI's official PyTorch
ROCm container plus a small overlay (`user-software.squashfs`) containing the
extra Python packages we need on top — currently just `math_verify` and its
dep `latex2sympy2_extended`. Everything stays under `containers/easybuild/`;
nothing is written to `~/.local`, `~/.cache`, or any shared project space.

## Files

| File | Purpose |
|------|---------|
| `build_qwen36.sh` | Idempotent build script. Runs EasyBuild once, creates the venv, runs `pip install`, bakes the result into a squashfs, verifies imports. |
| `qwen36_requirements.txt` | The pip packages that get added on top of the conda env in the LUMI SIF. |
| `easybuild/` | Private `EBU_USER_PREFIX`. Contains the SIF, generated module, and the squashfs overlay. ~18 GB (the SIF dominates). |

## How it works

LUMI ships an EasyBuild recipe for the AMD-built PyTorch ROCm container.
Loading the resulting `PyTorch/2.7.1-rocm-6.2.4-python-3.12-singularity-20250827`
module exposes a Singularity container with PyTorch + a `pytorch` conda env
(includes `transformers 4.55.3`, `accelerate 0.34.2`, `vllm`, `flash-attn`,
etc.) and provides wrapper scripts (`python`, `pip`, `accelerate`,
`huggingface-cli`, `torchrun`, …) that transparently `singularity exec` into
the container.

The canonical way to add extra packages without modifying the SIF is a
*writable Python virtualenv overlay*:

1. EasyBuild installs the SIF and a Lmod module under our `EBU_USER_PREFIX`.
2. The recipe's `postinstallcmds` create an empty venv at
   `$CONTAINERROOT/user-software/venv/pytorch` (with `--system-site-packages`
   so it inherits the conda env's torch/transformers/etc.) — done by an
   explicit `singularity exec --bind <user-software>:/user-software ...`.
3. Module's Lmod `create_container_vars` (defined in
   `/appl/lumi/LUMI-SoftwareStack/LMOD/SitePackage.lua:685`) decides what to
   mount at `/user-software` inside the container at load time:
   - if `$CONTAINERROOT/user-software.squashfs` exists → mounted **read-only**
     (`image-src=/`)
   - else if `$CONTAINERROOT/user-software/` directory exists → mounted
     **writable**
   - else → no `/user-software` bind at all.
4. `pip install` from the host (via the `pip` wrapper) writes through the
   writable bind into `/user-software/venv/pytorch/lib/python3.12/site-packages`.
5. `make-squashfs` (provided by the module) packs `user-software/` into
   `user-software.squashfs`. The dir is then redundant; we delete it.
6. After reloading the module, `/user-software` is mounted from the squashfs
   (much friendlier on Lustre than thousands of small files).

## Run it

```bash
bash containers/build_qwen36.sh
```

Idempotent: if the SIF + module are already installed, it just skips eb. The
post-install steps (cleanup → recreate venv → pip install → squash → verify)
run every time, so re-running the script after editing
`qwen36_requirements.txt` rebuilds the squashfs from scratch.

To use the container in a job:

```bash
sbatch sbatch/generate_traces_27b.sbatch
```

The sbatch's `module load` sequence mirrors the build script and locates the
private module by globbing `$EBU_USER_PREFIX/modules/**/PyTorch/*.lua`. The
PyTorch module ships `python`, `pip`, etc. wrappers that auto-`singularity
exec` into the container, so user code just calls `srun python …`.

## Non-obvious gotchas (read before debugging)

- **The wrapper-script `python -m venv` does NOT work for creating the
  initial venv.** The wrappers `singularity exec` without binding
  `user-software/` writable, so the venv ends up in a discarded tmpfs layer.
  Always use the canonical
  `singularity exec --bind $CONTAINERROOT/user-software:/user-software <SIF>
  bash -c '$WITH_CONDA ; cd /user-software/venv ; python -m venv
  --system-site-packages pytorch'` pattern.

- **A leftover empty `user-software.squashfs` is the most common cause of
  pip falling back to `~/.local`.** Its presence flips the bind to
  read-only, so pip can't write into the venv. The `Defaulting to user
  installation because normal site-packages is not writeable` message is
  the symptom. The build script removes any prior squashfs at the start.

- **Module reload after deleting the squashfs.** Lmod evaluates
  `SINGULARITY_BIND` at load time, so if you delete the squashfs *after*
  the module is loaded, every `singularity exec` keeps trying to bind a
  now-missing file. The build script does
  `module purge -f && module load LUMI && module load PyTorch/...` after
  cleanup so `create_container_vars` runs against the fresh on-disk state.

- **`PYTHONNOUSERSITE=1` and `SINGULARITYENV_PYTHONNOUSERSITE=1`** are set
  by the build script so that even if pip's venv detection were to fail for
  some reason, Python would still refuse to load packages from `~/.local`.
  Belt and suspenders.

- **`--skip-sanity-check` on `eb`** is required because the recipe's sanity
  step pins exact versions of `transformers`, `vllm`, etc. that may not
  match what the public LUMI SIF actually ships. Our own post-install
  asserts at the end of `build_qwen36.sh` cover the symbols we care about.

- **Most packages from `qwen36_requirements.txt` are already in the conda
  env.** `transformers>=4.50` is satisfied by the conda's 4.55.3, etc., so
  pip skips them. Only `math_verify` (and its dep `latex2sympy2_extended`)
  end up in the squashfs. Verify after each rebuild that
  `pip show math_verify` reports a `/user-software/...` location, not
  `/opt/miniconda3/...` or `/users/<you>/.local/...`.

- **The private module lives at
  `easybuild/modules/container/PyTorch/<version>.lua`**, not under
  `easybuild/modules/LUMI/<ver>/partition/<L|G>/PyTorch/`. This is because
  `build_qwen36.sh` loads `partition/container` (the dummy "container"
  partition) before running `eb`, per LUMI doc convention. After
  `module load LUMI`, that path is on `MODULEPATH` automatically — no need
  to load `partition/container` again at job time.

## Rebuild from scratch

```bash
# Wipe the entire private prefix (incl. the SIF — ~18 GB redownload via eb).
rm -rf containers/easybuild
bash containers/build_qwen36.sh
```

To rebuild *just* the squashfs (keep the SIF):

```bash
CR=containers/easybuild/SW/container/PyTorch/2.7.1-rocm-6.2.4-python-3.12-singularity-20250827
rm -f  "$CR/user-software.squashfs"
rm -rf "$CR/user-software"
bash containers/build_qwen36.sh
```

## Verifying

After a successful build, the script prints a `[POST-SQUASH]` line with
torch / transformers / accelerate versions and a `math_verify.verify()`
truth check, all done after reloading the module so the squashfs is the
active mount. If the script exits 0, the container is ready.

To re-verify later from a fresh shell:

```bash
unset CONTAINERROOT SIF SIFPYTORCH SINGULARITY_BIND
export EBU_USER_PREFIX=$PWD/containers/easybuild
module purge -f
module load LUMI
module load PyTorch/2.7.1-rocm-6.2.4-python-3.12-singularity-20250827
python -c "import torch, transformers, math_verify; \
           print(torch.__version__, transformers.__version__); \
           from math_verify import parse, verify; \
           print(verify(parse(r'\boxed{42}'), parse(r'\boxed{42}')))"
```

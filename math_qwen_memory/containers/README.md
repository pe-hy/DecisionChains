# Private LUMI PyTorch container with extra pip packages

`build_qwen36.sh` produces a project-local copy of LUMI's official PyTorch
ROCm container plus an overlay (`user-software.squashfs`) containing extra
Python packages we need on top of the SIF's conda env. As of 2026-04 the
overlay carries `transformers 5.6.2`, `flash-linear-attention 0.5.0`,
`accelerate 1.13.0`, `tokenizers 0.22.2`, `safetensors 0.7.0`,
`huggingface-hub 1.12.0`, `math_verify 0.9.0` (+ deps). Everything stays
under `containers/easybuild/`; nothing is written to `~/.local`, `~/.cache`,
or any shared project space.

This README is intended to be **complete enough that a future Claude
instance can re-derive the floors in `qwen36_requirements.txt` and rebuild
the container from scratch when LUMI's SIF changes or new model
architectures need new fast-path libraries.** Read all of it before
editing.

## Files

| File | Purpose |
|------|---------|
| `build_qwen36.sh` | Idempotent build script. Runs EasyBuild once, creates the venv, runs `pip install`, bakes the result into a squashfs, verifies imports + Qwen3.6 fast-path symbols. |
| `qwen36_requirements.txt` | The pip floors. Each floor MUST be strictly above what the SIF's conda env ships, or pip silently skips. See **The floor rule** below. |
| `easybuild/` | Private `EBU_USER_PREFIX`. Contains the SIF, generated module, and the squashfs overlay. ~18 GB (the SIF dominates). Gitignored. |

## How it works

LUMI ships an EasyBuild recipe for the AMD-built PyTorch ROCm container
under `/appl/local/containers/LUMI-EasyBuild-containers/easybuild/easyconfigs/p/PyTorch/`.
Loading the resulting `PyTorch/<version>` module exposes a Singularity
container with PyTorch + a `pytorch` conda env (preloaded with transformers,
accelerate, vllm, flash-attn, causal-conv1d, …) and provides wrapper
scripts (`python`, `pip`, `accelerate`, `huggingface-cli`, `torchrun`, …)
that transparently `singularity exec` into the container.

The canonical way to add extra packages without modifying the SIF is a
*writable Python virtualenv overlay*:

1. EasyBuild installs the SIF and a Lmod module under our `EBU_USER_PREFIX`.
   The recipe's `postinstallcmds` create an empty venv at
   `$CONTAINERROOT/user-software/venv/pytorch` (with `--system-site-packages`
   so it inherits the conda env's torch/transformers/etc.) — done by an
   explicit `singularity exec --bind <user-software>:/user-software ...`.
2. The Lmod function `create_container_vars` (defined in
   `/appl/lumi/LUMI-SoftwareStack/LMOD/SitePackage.lua:685`) decides what to
   mount at `/user-software` inside the container at module-load time:
   - if `$CONTAINERROOT/user-software.squashfs` exists → mounted **read-only**
     (`image-src=/`)
   - else if `$CONTAINERROOT/user-software/` directory exists → mounted
     **writable**
   - else → no `/user-software` bind at all.
3. `pip install` from the host (via the wrapper script) runs inside the
   container and writes through the writable bind into
   `/user-software/venv/pytorch/lib/python3.12/site-packages`.
4. `make-squashfs` (provided by the module) packs the dir into
   `user-software.squashfs`. The dir is then redundant; we delete it.
5. After reloading the module, `/user-software` is mounted from the
   squashfs (much friendlier on Lustre than thousands of small files).

## The floor rule (read before editing requirements.txt)

The venv is created with `--system-site-packages`, so packages already
installed in the SIF's conda env are visible to Python inside the venv. If
a requirement floor in `qwen36_requirements.txt` is satisfied by the conda
version, **pip prints "Requirement already satisfied" and writes nothing
into the squashfs**. The slurm job then runs against the conda version, not
the version you intended.

Concrete failure mode that took us multiple cycles to diagnose: writing
`transformers>=4.50` while conda ships 4.55.3. Pip skipped, the squashfs
shipped without a newer transformers, and the slurm job died with
`KeyError: 'qwen3_5'` because conda's 4.55.3 predates Qwen3.6 support.

**Therefore: every floor must be strictly above the conda version.** The
build script enforces this — after pip install, it asserts that every
named package in `qwen36_requirements.txt` resolves to a path under
`/user-software/`, and aborts before make-squashfs if any escaped to
`/opt/miniconda3/`.

To discover what conda ships before bumping floors:

```bash
# Load the module, then pip-list inside the container:
unset CONTAINERROOT SIF SIFPYTORCH SINGULARITY_BIND
export EBU_USER_PREFIX=$PWD/containers/easybuild
module purge -f
module load LUMI
module load PyTorch/<version>
singularity exec $SIFPYTORCH pip list 2>/dev/null \
    | grep -iE 'transformers|accelerate|tokeniz|safetens|huggingface|fla|conv1d|flash'
```

Bump each floor in `qwen36_requirements.txt` to **strictly greater than**
the conda version (e.g. conda 0.6.2 → write `>=0.7`).

Compare to PyPI before committing:

```bash
# What's the latest available?
singularity exec $SIFPYTORCH pip index versions <package>
```

When a package is already in conda at a version newer than what you'd
bump to, you don't need to list it at all — let it resolve through
`--system-site-packages`. (That's how `causal-conv1d 1.5.2` is used:
listed nowhere in `qwen36_requirements.txt`, used by Qwen3.6's fast path
via the conda inheritance.)

## Discovering fast-path / kernel-library requirements for new architectures

If a future model architecture (`qwen3_X`, `mamba3`, etc.) requires GPU
kernel libraries beyond what conda ships, the diagnostic pattern is:

1. **Read the modeling file inside the squashfs.** The transformers
   convention is to define `is_fast_path_available = all((sym1, sym2,
   ...))` near the top of `models/<model_type>/modeling_<model_type>.py`,
   where each `symN` is conditionally imported from a kernel library and
   set to `None` if missing.

   Find it:
   ```bash
   singularity exec $SIFPYTORCH bash -c "
     python -c '
       import transformers, os, glob
       p = os.path.dirname(transformers.__file__)
       for f in sorted(glob.glob(p + \"/models/<model_type>/*.py\")):
           print(f)
       '"
   # then read the imports + the is_fast_path_available expression
   ```
2. **Probe each kernel library on PyPI.** Pure-Python wheels
   (`*-py3-none-any.whl`) install cleanly; sdist-only packages compile
   from source and need ROCm toolchain (hipcc + amdclang are inside the
   SIF; gcc is **not**, so set `CC=/opt/rocm-*/bin/amdclang
   CXX=/opt/rocm-*/bin/amdclang++` and `PYTORCH_ROCM_ARCH=gfx90a` for
   MI250X).
3. **Check whether the kernel library is already in conda.** Many of the
   common ones (`causal-conv1d`, `flash-attn`, `xformers`, `vllm`) are
   pre-built for ROCm in the LUMI SIF and just need to be made visible
   to a NEWER transformers via `--system-site-packages`. Don't rebuild
   what's already there.
4. **Add an assertion to `build_qwen36.sh`.** Replicate the
   `is_fast_path_available = all((...))` expression at build time so a
   missing/broken kernel fails the build instead of silently degrading
   to the torch fallback at job time. Current code does this for
   Qwen3.6's 4-symbol check (`build_qwen36.sh` post-squash python
   block).

## Run it

```bash
bash containers/build_qwen36.sh
```

Idempotent: if the SIF + module are already installed, it just skips eb.
The post-install steps (cleanup → recreate venv → pip install → squash →
verify) run every time, so re-running the script after editing
`qwen36_requirements.txt` rebuilds the squashfs from scratch.

To use the container in a job:

```bash
sbatch sbatch/generate_traces_27b.sbatch
```

The sbatch's `module load` sequence mirrors the build script and locates
the private module by globbing `$EBU_USER_PREFIX/modules/**/PyTorch/*.lua`.
The PyTorch module ships `python`, `pip`, etc. wrappers that
auto-`singularity exec` into the container, so user code just calls
`srun python …`.

## Non-obvious gotchas (read before debugging)

- **The floor rule** (above) is the most common silent failure. If the
  slurm job dies with a `KeyError` on `model_type`, or a model loads but
  prints "fast path is not available" / "this feature requires
  transformers>=X.Y", check whether the package actually landed in the
  squashfs:
  ```bash
  singularity exec $SIFPYTORCH pip show <pkg> | awk '/^Location:/ {print $2}'
  # /user-software/...   = OK, in squashfs
  # /opt/miniconda3/...  = pip skipped, bump the floor
  ```

- **The wrapper-script `python -m venv` does NOT work for creating the
  initial venv.** The wrappers `singularity exec` without binding
  `user-software/` writable, so the venv ends up in a discarded tmpfs
  layer. Always use the canonical
  `singularity exec --bind $CONTAINERROOT/user-software:/user-software
  <SIF> bash -c '$WITH_CONDA ; cd /user-software/venv ; python -m venv
  --system-site-packages pytorch'` pattern (matches the easyconfig's own
  `postinstallcmds`).

- **A leftover `user-software.squashfs` flips `/user-software` to
  read-only at module-load time** (`SitePackage.lua:705-720`), causing
  pip to fall back to `~/.local` with the message *"Defaulting to user
  installation because normal site-packages is not writeable"*. The
  build script removes any prior squashfs at the start.

- **Module reload after deleting the squashfs** is mandatory. Lmod
  evaluates `SINGULARITY_BIND` at load time, so if you delete the
  squashfs *after* the module is loaded, every subsequent
  `singularity exec` keeps trying to bind the now-missing file with
  `failed to load data image .../user-software.squashfs: no such file`.
  The build script does `module purge -f && module load LUMI && module
  load PyTorch/...` after cleanup so `create_container_vars` runs
  against the fresh on-disk state.

- **`PYTHONNOUSERSITE=1` and `SINGULARITYENV_PYTHONNOUSERSITE=1`** are
  set by the build script so that even if pip's venv detection were to
  fail for some reason, Python would still refuse to load packages from
  `~/.local`. Belt and suspenders.

- **`--skip-sanity-check` on `eb`** is required because the recipe's
  sanity step pins exact versions of `transformers`, `vllm`, etc. that
  the public LUMI SIF doesn't always match. Our own post-install
  asserts at the end of `build_qwen36.sh` cover the symbols we care
  about.

- **The pip name and import name often differ.** `flash-linear-attention`
  imports as `fla`. `huggingface-hub` imports as `huggingface_hub`. The
  build script's escape-check loop iterates the import-name forms (so
  `pip show` works on both — `pip show` accepts either the dist name or
  the canonical underscore form).

- **The private module lives at
  `easybuild/modules/container/PyTorch/<version>.lua`**, not under
  `easybuild/modules/LUMI/<ver>/partition/<L|G>/PyTorch/`. This is
  because `build_qwen36.sh` loads `partition/container` (the dummy
  "container" partition) before running `eb`, per LUMI doc convention.
  After `module load LUMI`, that path is on `MODULEPATH` automatically
  — no need to load `partition/container` again at job time. The sbatch
  finds the module via
  `find "$EBU_USER_PREFIX/modules" -name '[0-9]*.lua' | grep
  '/PyTorch/...-singularity-...\.lua$'`.

- **Compiling from source on the login node** works for ROCm via
  `hipcc` + `amdclang++` (both at `/opt/rocm-6.2.4/bin/...`). There is
  **no** `gcc` / `g++` inside the SIF, so set `CC` and `CXX` to the
  amdclang paths before pip install if a package builds C++ extensions.
  Login nodes do not have a GPU, but kernel compilation only needs the
  ROCm toolchain (not a runtime device). For MI250X targets export
  `PYTORCH_ROCM_ARCH=gfx90a`.

- **Login-node Triton warnings are expected.** Build verification on
  the login node prints `Triton is not supported on current platform,
  roll back to CPU.` because there is no GPU. On the compute node
  (small-g / standard-g, partition/G) Triton-ROCm activates and the
  fla / qwen3_5 kernels JIT-compile on first call. The build script
  treats this as a non-error.

## When the SIF version updates

LUMI rolls new SIFs every few months. To migrate:

1. Run `bash containers/build_qwen36.sh` — it picks the newest recipe
   from `/appl/local/containers/LUMI-EasyBuild-containers/easybuild/easyconfigs/p/PyTorch/`
   automatically and installs to a versioned path under
   `easybuild/SW/container/PyTorch/<new-version>/`.
2. Run the conda-snapshot probe (in **The floor rule** above) and
   compare against the floors in `qwen36_requirements.txt`. If the new
   SIF's conda already ships a newer version of one of our packages,
   bump the floor strictly above it.
3. The build script's hard-assert loop catches anything that escapes
   into conda; let it abort and tell you which package needs a higher
   floor.
4. Re-run the build, confirm `[POST-SQUASH]` prints all green, then
   resubmit the sbatch.

## Rebuild from scratch

```bash
# Wipe the entire private prefix (incl. the SIF — ~18 GB redownload via eb).
rm -rf containers/easybuild
bash containers/build_qwen36.sh
```

To rebuild *just* the squashfs (keep the SIF):

```bash
# build_qwen36.sh already does this automatically — just re-run it.
bash containers/build_qwen36.sh
```

## Verifying

After a successful build, the script prints `[POST-SQUASH]` lines with
torch / transformers / accelerate / fla versions, a `math_verify.verify()`
truth check, an `AutoConfig.from_pretrained` smoke test on the actual
Qwen3.6-27B `config.json` from `hf_cache/`, and a 4-symbol
`is_fast_path_available` assertion — all done after reloading the module
so the squashfs is the active mount. If the script exits 0 with all those
lines green, the container is ready.

To re-verify later from a fresh shell:

```bash
unset CONTAINERROOT SIF SIFPYTORCH SINGULARITY_BIND
export EBU_USER_PREFIX=$PWD/containers/easybuild
module purge -f
module load LUMI
module load PyTorch/$(ls easybuild/modules/container/PyTorch/ | sed 's/\.lua$//' | head -1)
python <<'PY'
import torch, transformers, math_verify, fla, importlib.metadata as m
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule, fused_recurrent_gated_delta_rule,
)
from math_verify import parse, verify
print('torch:', torch.__version__, 'hip:', torch.version.hip)
print('transformers:', transformers.__version__)
print('flash-linear-attention:', m.version('flash-linear-attention'))
print('causal-conv1d:', m.version('causal-conv1d'))
print('math_verify(42,42):', verify(parse(r'\boxed{42}'), parse(r'\boxed{42}')))
print('fast_path:', all((causal_conv1d_fn, causal_conv1d_update,
                         chunk_gated_delta_rule, fused_recurrent_gated_delta_rule)))
PY
```

Expected output (all five lines, fast_path=True) means the container is
healthy.

#!/usr/bin/env bash
# Build a private LUMI PyTorch container with extra pip packages baked into
# user-software.squashfs.  Everything stays under containers/easybuild — never
# touches ~/.local, ~/.cache, or the project_465002050 shared overlay.
#
# Refs:
#   https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/p/PyTorch/
#   https://docs.lumi-supercomputer.eu/software/installing/easybuild/
#   /appl/lumi/LUMI-SoftwareStack/LMOD/SitePackage.lua  (create_container_vars)

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REQS="$HERE/qwen36_requirements.txt"

# ---------------------------------------------------------------------------
# 1) Pin every writable location to project-local. Nothing must escape.
#    EBU_USER_PREFIX MUST be set before any `module load` (LUMI doc warning:
#    changing it after a LUMI module is loaded has side effects).
# ---------------------------------------------------------------------------
PRIVATE_EBU_PREFIX="$HERE/easybuild"
export EBU_USER_PREFIX="$PRIVATE_EBU_PREFIX"
mkdir -p "$EBU_USER_PREFIX"

# Bar pip from ever falling back to ~/.local.  Both on the host and inside the
# singularity container (the SINGULARITYENV_* vars are forwarded).
export PYTHONNOUSERSITE=1
export PIP_USER=0
export PIP_CACHE_DIR="$HERE/.pipcache"
export SINGULARITYENV_PYTHONNOUSERSITE=1
export SINGULARITYENV_PIP_USER=0
export SINGULARITYENV_PIP_CACHE_DIR=/tmp/pipcache  # writable inside container
mkdir -p "$PIP_CACHE_DIR"

# ---------------------------------------------------------------------------
# 2) Reset module + container env from any prior shell state.
# ---------------------------------------------------------------------------
unset CONTAINERROOT SIF SIFPYTORCH SINGULARITY_BIND
module purge -f 2>/dev/null || true

# ---------------------------------------------------------------------------
# 3) Pick the newest mainline PyTorch container easyconfig.
# ---------------------------------------------------------------------------
LUMI_PT_RECIPES=/appl/local/containers/LUMI-EasyBuild-containers/easybuild/easyconfigs/p/PyTorch
NEWEST_EB=$(ls "$LUMI_PT_RECIPES"/PyTorch-*-rocm-*-python-*-singularity-*.eb 2>/dev/null \
    | grep -E 'PyTorch-[0-9]+\.[0-9]+\.[0-9]+-rocm-.*-singularity-[0-9]{8}\.eb$' \
    | sort -r | head -1)
[ -z "$NEWEST_EB" ] && { echo "ERROR: no PyTorch easyconfig under $LUMI_PT_RECIPES" >&2; exit 1; }
EB_FILE=$(basename "$NEWEST_EB")
PYTORCH_MODULE="PyTorch/${EB_FILE#PyTorch-}"
PYTORCH_MODULE="${PYTORCH_MODULE%.eb}"

echo "[build] EBU_USER_PREFIX = $EBU_USER_PREFIX"
echo "[build] target module   = $PYTORCH_MODULE"
echo "[build] eb recipe       = $EB_FILE"

# ---------------------------------------------------------------------------
# 4) Install via EasyBuild only if not already installed under our prefix.
# ---------------------------------------------------------------------------
module load LUMI
if ! module load "$PYTORCH_MODULE" 2>/dev/null; then
    echo "[build] $PYTORCH_MODULE not yet installed — running eb…"
    module load partition/container
    module load EasyBuild-user
    # --skip-sanity-check: the recipe's sanity asserts pin exact versions of
    # transformers / vllm / huggingface-cli that the public LUMI SIF doesn't
    # always match.  We run our own asserts after install instead.
    eb "$EB_FILE" ${EB_FLAGS:---skip-sanity-check}
    module purge -f 2>/dev/null || true
    module load LUMI
fi

# Find the partition the module was installed under (typically L for login).
PT_PARTITION=$(find "$EBU_USER_PREFIX/modules/LUMI/" -name "${PYTORCH_MODULE#PyTorch/}.lua" 2>/dev/null \
    | head -1 | sed -nE 's|.*/partition/([^/]+)/PyTorch/.*|\1|p')
[ -n "$PT_PARTITION" ] && module load "partition/$PT_PARTITION"
module load "$PYTORCH_MODULE"

[ -z "${CONTAINERROOT:-}" ] && { echo "ERROR: CONTAINERROOT not set after module load" >&2; exit 1; }
[ -z "${SIFPYTORCH:-}" ]    && { echo "ERROR: SIFPYTORCH not set after module load" >&2; exit 1; }

# Capture SIF path now so we still have it after another module purge.
SIF_LOCAL="$SIFPYTORCH"
echo "[build] CONTAINERROOT = $CONTAINERROOT"
echo "[build] SIF           = $SIF_LOCAL"

# ---------------------------------------------------------------------------
# 5) Cleanup ANY stale state from prior partial builds:
#    - Empty user-software/  (left by previous build script attempt)
#    - Empty user-software.squashfs  (eb created venv → make-squashfs ran on
#      empty venv → squashfs is RO bound → pip can't write).
#    Once both are gone, the venv binding logic in SitePackage.lua falls
#    through to "no /user-software bind" — clean slate.
# ---------------------------------------------------------------------------
echo "[clean] wiping any stale user-software state…"
rm -f  "$CONTAINERROOT/user-software.squashfs"
rm -rf "$CONTAINERROOT/user-software"

# After deleting the stale squashfs, the SINGULARITY_BIND that was set by the
# previous `module load PyTorch/...` still references the now-gone file.  We
# must purge & reload so SitePackage.lua re-evaluates the bind list from the
# fresh on-disk state (no squashfs, no user-software dir → no /user-software
# bind at all).  Otherwise `singularity exec` aborts with
#   "failed to load data image .../user-software.squashfs: no such file".
module purge -f 2>/dev/null || true
module load LUMI
[ -n "$PT_PARTITION" ] && module load "partition/$PT_PARTITION"
module load "$PYTORCH_MODULE"

# ---------------------------------------------------------------------------
# 6) Re-create the venv EXACTLY the way the easyconfig's postinstallcmds does:
#       singularity exec --bind <user-software>:/user-software <SIF> \
#           bash -c '$WITH_CONDA ; cd /user-software/venv ; \
#                    python -m venv --system-site-packages pytorch'
#
#    Crucially we DON'T use the `python` wrapper here — the wrapper's
#    singularity exec doesn't bind user-software writable, so the venv would
#    end up empty (the bug we're fixing).
# ---------------------------------------------------------------------------
mkdir -p "$CONTAINERROOT/user-software/venv"
echo "[venv] creating /user-software/venv/pytorch (via direct singularity exec)…"
singularity exec \
    --bind "$CONTAINERROOT/user-software:/user-software" \
    "$SIF_LOCAL" \
    bash -c '${WITH_CONDA:-true} ; cd /user-software/venv ; python -m venv --system-site-packages pytorch'

if [ ! -f "$CONTAINERROOT/user-software/venv/pytorch/pyvenv.cfg" ]; then
    echo "ERROR: venv creation failed — pyvenv.cfg not found." >&2
    exit 1
fi
echo "[venv] OK: $(ls "$CONTAINERROOT/user-software/venv/pytorch/bin" | wc -l) entries in venv/bin/"

# ---------------------------------------------------------------------------
# 7) Reload module so SitePackage.lua re-evaluates create_container_vars.
#    With user-software/ now a directory and no squashfs file present, the
#    Lua chooses the "writable directory" branch → /user-software is RW
#    inside the container → pip can write into the venv.
# ---------------------------------------------------------------------------
module purge -f 2>/dev/null || true
module load LUMI
[ -n "$PT_PARTITION" ] && module load "partition/$PT_PARTITION"
module load "$PYTORCH_MODULE"

PYBIN=$(which python)
PIPBIN=$(which pip)
echo "[reload] python -> $PYBIN"
echo "[reload] pip    -> $PIPBIN"
case "$PYBIN" in
    "$CONTAINERROOT/"*) ;;
    *) echo "ERROR: python wrapper not under \$CONTAINERROOT" >&2; exit 1 ;;
esac

# Quick sanity: torch is ROCm.
python -c "
import torch
assert torch.version.hip is not None, 'base torch is not ROCm — wrong module?'
print('[sanity] torch:', torch.__version__, 'hip:', torch.version.hip)
"

# Confirm pip is now resolving INSIDE the container to the venv's pip.
python - <<'PY'
import sys, os
print('[sanity] sys.prefix:', sys.prefix)
print('[sanity] VIRTUAL_ENV:', os.environ.get('VIRTUAL_ENV'))
assert sys.prefix == '/user-software/venv/pytorch', f'venv not active: {sys.prefix}'
PY

# ---------------------------------------------------------------------------
# 8) Install our extra packages.  --no-cache-dir to avoid using ~/.cache.
# ---------------------------------------------------------------------------
echo "[pip] installing $REQS into /user-software/venv/pytorch …"
pip install --no-cache-dir -r "$REQS"

# ---------------------------------------------------------------------------
# 9) Verify install location.  Our floors in qwen36_requirements.txt are set
#    ABOVE what the conda env ships, so transformers/accelerate/tokenizers/
#    safetensors/huggingface_hub MUST end up in the venv.  If any of them
#    still resolve to /opt/miniconda3/... pip silently skipped — abort.
# ---------------------------------------------------------------------------
echo "[pip] post-install locations:"
for pkg in transformers accelerate tokenizers safetensors huggingface_hub math_verify flash_linear_attention; do
    loc=$(pip show "$pkg" 2>/dev/null | awk '/^Location:/ {print $2}')
    printf '    %-18s -> %s\n' "$pkg" "${loc:-<missing>}"
done

# Hard assert: every package that we want NEWER than conda must now live in
# /user-software.  If any escaped (e.g. pip thought conda's was good enough)
# it won't be in the squashfs and the slurm job will fail with the same
# "model_type=qwen3_5 unknown" error we just hit.
escaped=()
for pkg in transformers tokenizers accelerate safetensors huggingface_hub math_verify flash_linear_attention; do
    loc=$(pip show "$pkg" 2>/dev/null | awk '/^Location:/ {print $2}')
    case "$loc" in
        */user-software/*) ;;  # OK
        *) escaped+=("$pkg=$loc") ;;
    esac
done
if [ ${#escaped[@]} -gt 0 ]; then
    echo "ERROR: these packages did NOT land in /user-software:" >&2
    printf '    %s\n' "${escaped[@]}" >&2
    echo "       Bump the floor in qwen36_requirements.txt above the version" >&2
    echo "       reported above so pip is forced to upgrade into the venv." >&2
    exit 1
fi
echo "[pip] OK: all required packages installed under /user-software"

# Last gate: full import + symbol smoke test before squashing, plus the
# critical AutoConfig.from_pretrained on Qwen3.6-27B's config (the actual
# breakage we are fixing — earlier 4.55.3 raised KeyError('qwen3_5')).
python - <<'PY'
import os, glob, torch, transformers, accelerate, tokenizers, safetensors, huggingface_hub
print('[pre-squash] torch:', torch.__version__, 'hip:', torch.version.hip)
print('[pre-squash] transformers:', transformers.__version__)
print('[pre-squash] accelerate:', accelerate.__version__)
print('[pre-squash] tokenizers:', tokenizers.__version__)
print('[pre-squash] safetensors:', safetensors.__version__)
print('[pre-squash] huggingface_hub:', huggingface_hub.__version__)

from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer,
    LogitsProcessorList, TemperatureLogitsWarper,
    TopPLogitsWarper, TopKLogitsWarper, MinPLogitsWarper,
)
print('[pre-squash] transformers symbols: OK')

from math_verify import parse, verify
print('[pre-squash] math_verify: OK; parse(\\\\boxed{42}) =', parse(r'\boxed{42}'))

# Critical assertion: load the actual Qwen3.6-27B config we will be using
# at job time.  If the model is not yet in hf_cache, skip with a warning.
hf_cache = os.environ.get('HF_HUB_CACHE') or '/pfs/lustrep4/scratch/project_465002631/Petr/DecisionChains/math_qwen_memory/hf_cache'
candidates = glob.glob(f'{hf_cache}/models--Qwen--Qwen3.6-27B/snapshots/*/config.json')
if candidates:
    snap_dir = os.path.dirname(candidates[0])
    cfg = AutoConfig.from_pretrained(snap_dir, trust_remote_code=False)
    print(f'[pre-squash] AutoConfig.from_pretrained(Qwen3.6-27B): model_type={cfg.model_type!r} '
          f'arch={cfg.architectures!r} OK')
else:
    print(f'[pre-squash] (skip: no Qwen3.6-27B snapshot under {hf_cache})')

# Same check for Gemma 4 26B-A4B-it (model_type="gemma4", needs transformers>=5.5).
g4_candidates = glob.glob(f'{hf_cache}/models--google--gemma-4-26B-A4B-it/snapshots/*/config.json')
if g4_candidates:
    snap_dir = os.path.dirname(g4_candidates[0])
    cfg = AutoConfig.from_pretrained(snap_dir, trust_remote_code=False)
    print(f'[pre-squash] AutoConfig.from_pretrained(Gemma-4-26B-A4B-it): model_type={cfg.model_type!r} '
          f'arch={cfg.architectures!r} OK')
else:
    print(f'[pre-squash] (skip: no Gemma-4-26B-A4B-it snapshot under {hf_cache})')

# Qwen3.6 fast-path symbol availability.
# Note: pip pkg is `flash-linear-attention`, importable as `fla`.
import fla, importlib.metadata
print('[pre-squash] flash-linear-attention:',
      importlib.metadata.version('flash-linear-attention'),
      'fla.__file__:', fla.__file__)
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule, fused_recurrent_gated_delta_rule,
)
fast = all((causal_conv1d_fn, causal_conv1d_update,
            chunk_gated_delta_rule, fused_recurrent_gated_delta_rule))
assert fast, 'Qwen3.6 fast path NOT available before squash'
print('[pre-squash] Qwen3.6 fast path: AVAILABLE')
PY

# ---------------------------------------------------------------------------
# 10) Bake into squashfs.  The make-squashfs wrapper refuses to overwrite, but
#     we removed the old one in step 5 already.
# ---------------------------------------------------------------------------
echo "[squashfs] running make-squashfs…"
make-squashfs

# After squashing, the writable directory is redundant.  Leave a tiny .keep
# placeholder?  No — the Lua picks squashfs over directory anyway.  But to
# avoid any future shadowing surprises, remove the dir.
rm -rf "$CONTAINERROOT/user-software"
echo "[squashfs] $CONTAINERROOT/user-software.squashfs ready (dir removed)."

# ---------------------------------------------------------------------------
# 11) Reload module → /user-software is now mounted read-only from squashfs.
#     Final import check.
# ---------------------------------------------------------------------------
module purge -f 2>/dev/null || true
module load LUMI
[ -n "$PT_PARTITION" ] && module load "partition/$PT_PARTITION"
module load "$PYTORCH_MODULE"

python - <<'PY'
import os, glob, torch, transformers, accelerate, math_verify
print('[POST-SQUASH] torch:', torch.__version__, 'hip:', torch.version.hip)
print('[POST-SQUASH] transformers:', transformers.__version__)
print('[POST-SQUASH] accelerate:', accelerate.__version__)

from math_verify import parse, verify
gold, pred = parse(r'\boxed{42}'), parse(r'\boxed{42}')
print('[POST-SQUASH] math_verify verify(42,42):', verify(gold, pred))

# Same Qwen3.6-27B config check, but now reading the squashfs-bound venv.
from transformers import AutoConfig
hf_cache = '/pfs/lustrep4/scratch/project_465002631/Petr/DecisionChains/math_qwen_memory/hf_cache'
candidates = glob.glob(f'{hf_cache}/models--Qwen--Qwen3.6-27B/snapshots/*/config.json')
if candidates:
    cfg = AutoConfig.from_pretrained(os.path.dirname(candidates[0]), trust_remote_code=False)
    assert cfg.model_type == 'qwen3_5', f'unexpected model_type {cfg.model_type!r}'
    print(f'[POST-SQUASH] Qwen3.6-27B AutoConfig: model_type={cfg.model_type!r} OK')
else:
    print('[POST-SQUASH] (skip Qwen3.6 config check: snapshot not present)')

g4_candidates = glob.glob(f'{hf_cache}/models--google--gemma-4-26B-A4B-it/snapshots/*/config.json')
if g4_candidates:
    cfg = AutoConfig.from_pretrained(os.path.dirname(g4_candidates[0]), trust_remote_code=False)
    assert cfg.model_type == 'gemma4', f'unexpected model_type {cfg.model_type!r}'
    print(f'[POST-SQUASH] Gemma-4-26B-A4B-it AutoConfig: model_type={cfg.model_type!r} OK')
else:
    print('[POST-SQUASH] (skip Gemma-4 config check: snapshot not present)')

# Qwen3.6 fast-path check.  modeling_qwen3_5.py builds is_fast_path_available
# from these four symbols:
#   causal_conv1d_fn, causal_conv1d_update           (causal-conv1d, in conda)
#   chunk_gated_delta_rule, fused_recurrent_gated_delta_rule  (fla, in venv)
# If any is None the model falls back to torch and generation is ~10x slower.
import fla, importlib.metadata
print('[POST-SQUASH] flash-linear-attention:',
      importlib.metadata.version('flash-linear-attention'),
      'fla.__file__:', fla.__file__)
print('[POST-SQUASH] causal-conv1d:',
      importlib.metadata.version('causal-conv1d'))
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from fla.modules import FusedRMSNormGated
from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule, fused_recurrent_gated_delta_rule,
)
fast = all((causal_conv1d_fn, causal_conv1d_update,
            chunk_gated_delta_rule, fused_recurrent_gated_delta_rule))
assert fast, 'Qwen3.6 fast path is NOT available — one of the 4 syms is None'
print('[POST-SQUASH] Qwen3.6 fast path: AVAILABLE (fla + causal-conv1d both wired)')
print('[POST-SQUASH] all imports OK from squashfs')
PY

cat <<EOF

[DONE] Container ready.
       artifact: $CONTAINERROOT/user-software.squashfs
       module:   $PYTORCH_MODULE
       prefix:   $EBU_USER_PREFIX

Run with:
    sbatch sbatch/generate_traces_27b.sbatch
EOF

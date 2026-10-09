# Shared setup for the run scripts; each one sources this rather than running it.
#
#   ONE_LAYER_DEEPER  your checkout of tilde-research/one-layer-deeper
#                     (default: the current directory)
#   PYTHON            the interpreter with that checkout's dependencies
#                     (default: its .venv/bin/python)
#
# Scripts run from inside the checkout, so generated data and derived manifests
# land in its data/generated/ and benchmark/manifests/, while the submissions,
# generators and base manifests are read from this repository.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ONE_LAYER_DEEPER:-.}" || exit 1
if [ ! -f benchmark/runner.py ]; then
  echo "Not a one-layer-deeper checkout: $PWD" >&2
  echo "Run from one, or set ONE_LAYER_DEEPER to its path." >&2
  exit 1
fi

PYTHON="${PYTHON:-.venv/bin/python}"
# The generators import the checkout's data package from outside it.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

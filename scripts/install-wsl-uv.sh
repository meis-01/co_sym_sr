#!/usr/bin/env bash
set -euo pipefail

apt_packages=(
  ca-certificates curl python3-dolfin python3-gmsh python3-meshio
  python3-h5py python3-scipy python3-matplotlib python3-pytest
)
missing_packages=()
for package in "${apt_packages[@]}"; do
  dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q 'install ok installed' || missing_packages+=("$package")
done
if ((${#missing_packages[@]})); then
  sudo apt-get update
  sudo apt-get install -y "${missing_packages[@]}"
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

uv venv --clear --python /usr/bin/python3 --system-site-packages
uv sync --group dev \
  --no-install-package numpy \
  --no-install-package scipy \
  --no-install-package h5py \
  --no-install-package matplotlib \
  --no-install-package meshio \
  --no-install-package pytest

uv run --no-sync python -c 'import dolfin, gmsh, h5py, meshio, numpy, petsc4py, scipy; print("CFD environment ready:", dolfin.__version__, "NumPy", numpy.__version__)'

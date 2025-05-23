set -e  # Exit immediately on any error

PROJ_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Detect & Set Compatible GCC for CUDA 11.8
for v in 11 10 9 8 7; do
  if command -v gcc-$v &> /dev/null && command -v g++-$v &> /dev/null; then
    export CC=$(command -v gcc-$v)
    export CXX=$(command -v g++-$v)
    echo "Using GCC version $v: $CC"
    break
  fi
done

if [ -z "$CC" ] || [ -z "$CXX" ]; then
  echo "No supported GCC version found. CUDA 11.8 requires GCC ≤ 11."
  exit 1
fi

# Install mycpp
cd ${PROJ_ROOT}/mycpp/ && \
rm -rf build && mkdir -p build && cd build && \
cmake .. && \
make -j$(nproc)

# Build mycuda
cd ${PROJ_ROOT}/bundlesdf/mycuda && \
rm -rf build *egg* *.so && \
PYTHONNOUSERSITE=1 python setup.py build_ext --inplace

cd ${PROJ_ROOT}
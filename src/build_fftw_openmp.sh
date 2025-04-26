#!/bin/bash

# ==== パラメータ ====
FFTW_VERSION="3.3.10"
INSTALL_PREFIX="/opt/fftw-openmp"
LLVM_PATH="/opt/homebrew/opt/llvm"

# ==== 依存チェック ====
set -e

echo "🔧 Setting environment..."
export PATH="$LLVM_PATH/bin:$PATH"
export CC="$LLVM_PATH/bin/clang"
export CXX="$LLVM_PATH/bin/clang++"
export CFLAGS="-fopenmp -O3"
export LDFLAGS="-L$LLVM_PATH/lib -fopenmp -lomp"
export CPPFLAGS="-I$LLVM_PATH/include"
export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"

# ==== ダウンロード & 展開 ====
echo "⬇️ Downloading FFTW ${FFTW_VERSION}..."
curl -LO "http://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz"
tar -xzf "fftw-${FFTW_VERSION}.tar.gz"
cd "fftw-${FFTW_VERSION}"

echo "🧼 Cleaning previous builds (if any)..."
make clean || true
# ==== コンフィグ & ビルド ====
echo "⚙️ Configuring..."
./configure --enable-float --enable-openmp --enable-neon --disable-fortran \
  --prefix="${INSTALL_PREFIX}" \
  CC="${CC}" \
  CXX="${CXX}" \
  CFLAGS="${CFLAGS}" \
  LDFLAGS="${LDFLAGS}" \
  CPPFLAGS="${CPPFLAGS}"

echo "🛠 Building..."
make -j"$(sysctl -n hw.ncpu)"

echo "📦 Installing to ${INSTALL_PREFIX} (sudo may be needed)..."
sudo make install

echo "✅ Done! FFTW with OpenMP installed at ${INSTALL_PREFIX}"

# syntax=docker/dockerfile:1

# The defaults use a small CPU-capable Ubuntu image. For CUDA builds, pass:
#   --build-arg DEV_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
#   --build-arg RUNTIME_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04
#   --build-arg PIXAL3D_ENABLE_CUDA=ON

ARG DEV_IMAGE=ubuntu:22.04
ARG RUNTIME_IMAGE=ubuntu:22.04
FROM ${DEV_IMAGE} AS dev

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        ninja-build \
        libpng-dev \
        libjpeg-dev \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
CMD ["bash"]

FROM dev AS cpp-build

ARG PIXAL3D_BUILD_TYPE=Release
ARG PIXAL3D_ENABLE_CUDA=OFF
ARG PIXAL3D_CUDA_TF32=OFF
ARG PIXAL3D_BUILD_TESTS=ON
ARG PIXAL3D_RUN_TESTS=ON

COPY . .

RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=${PIXAL3D_BUILD_TYPE} \
        -DBUILD_SHARED_LIBS=OFF \
        -DPIXAL3D_ENABLE_CUDA=${PIXAL3D_ENABLE_CUDA} \
        -DPIXAL3D_CUDA_TF32=${PIXAL3D_CUDA_TF32} \
        -DPIXAL3D_GGML_NATIVE=OFF \
        -DPIXAL3D_BUILD_CLI=ON \
        -DBUILD_TESTING=${PIXAL3D_BUILD_TESTS} \
    && cmake --build build --parallel \
    && if [ "${PIXAL3D_RUN_TESTS}" = "ON" ] && [ "${PIXAL3D_BUILD_TESTS}" = "ON" ] && [ "${PIXAL3D_ENABLE_CUDA}" != "ON" ]; then \
         (cd build && ctest --output-on-failure); \
       elif [ "${PIXAL3D_ENABLE_CUDA}" = "ON" ]; then \
         echo "CUDA build complete; run CTest in a container with --gpus all"; \
       fi

FROM ${RUNTIME_IMAGE} AS runtime

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
        libpng16-16 \
        libjpeg8 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=cpp-build /workspace/build/bin/pixal3d /usr/local/bin/pixal3d

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/pixal3d"]
CMD ["--help"]

# Build once, then load the identical image on dcn9 and dcn10.
# This is a source build of the vendored tree, not a release-wheel substitution.
ARG CUDA_BASE=nvidia/cuda:12.9.1-devel-ubuntu22.04
FROM ${CUDA_BASE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG PPD_PREPARED_BASE=0
# A locally retained toolchain/torch layer can resume an interrupted build.
RUN if [ "${PPD_PREPARED_BASE}" != "1" ]; then \
    apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-venv python3-pip git curl ca-certificates \
    build-essential pkg-config libnuma-dev libibverbs-dev librdmacm-dev \
    ibverbs-providers libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*; fi

RUN test -x /opt/ppd-runtime/bin/python || python3 -m venv /opt/ppd-runtime
ENV PATH="/opt/ppd-runtime/bin:${PATH}"

COPY vllm-source/requirements/ /tmp/vllm-requirements/
RUN python -m pip install pip==25.3 \
    && python -m pip install \
       --index-url https://download.pytorch.org/whl/cu129 \
       torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
COPY deploy/two_node/constraints.txt /opt/ppd-constraints.txt
ENV PIP_CONSTRAINT=/opt/ppd-constraints.txt \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=10 \
    PIP_RESUME_RETRIES=20
RUN python -m pip install \
       -r /tmp/vllm-requirements/build.txt -r /tmp/vllm-requirements/cuda.txt

WORKDIR /opt/ppd
COPY vllm-source/ /opt/ppd/vllm-source/
ARG PPD_MAX_JOBS=1
ARG PPD_NVCC_THREADS=1
ARG PPD_BUILD_DIR=/opt/ppd/build
ARG PPD_SKIP_FA3_BUILD=1
# Use a distinct argument name: base images may already set TORCH_CUDA_ARCH_LIST.
ARG PPD_CUDA_ARCH_LIST=8.9
# The version is a local project label. It does not assert upstream ABI identity.
# Some cached toolchain images omit the unversioned libnvrtc.so linker name.
# Use the matching torch CUDA wheel's library explicitly in that case as well.
RUN PPD_NVRTC_LIBRARY="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/cuda_nvrtc/lib/libnvrtc.so.12")')" \
    && test -r "${PPD_NVRTC_LIBRARY}" \
    && PPD_CUDA_INCLUDE_PATH="$(python -c 'import glob, sysconfig; print(":".join(sorted(glob.glob(sysconfig.get_path("purelib") + "/nvidia/*/include"))))')" \
    && CPATH="${PPD_CUDA_INCLUDE_PATH}${CPATH:+:${CPATH}}" \
    CMAKE_ARGS="-DCUDA_nvrtc_LIBRARY=${PPD_NVRTC_LIBRARY}" \
    VLLM_TARGET_DEVICE=cuda VLLM_USE_PRECOMPILED=0 \
    VLLM_VERSION_OVERRIDE=0.13.0+ppd.local \
    MAX_JOBS=${PPD_MAX_JOBS} NVCC_THREADS=${PPD_NVCC_THREADS} \
    VLLM_BUILD_DIR=${PPD_BUILD_DIR} \
    VLLM_SKIP_FA3_BUILD=${PPD_SKIP_FA3_BUILD} \
    TORCH_CUDA_ARCH_LIST=${PPD_CUDA_ARCH_LIST} \
    CMAKE_BUILD_TYPE=Release \
    python -m pip install -v --no-build-isolation --no-deps -e /opt/ppd/vllm-source

COPY requirements.txt /opt/ppd/requirements.txt
RUN python -m pip install -r /opt/ppd/requirements.txt \
    && python -m pip check \
    && python -m pip freeze --all > /opt/ppd/runtime-freeze.txt
# Do not advertise the cached base image's upstream release provenance.
ENV VLLM_USAGE_SOURCE=ppd-two-node-source \
    VLLM_BUILD_COMMIT=vendored-local \
    VLLM_BUILD_PIPELINE=local \
    VLLM_BUILD_URL="" \
    VLLM_IMAGE_TAG=ppd-qwen25:source-cu129
COPY ppd/ /opt/ppd/ppd/
COPY configs/ /opt/ppd/configs/
COPY scripts/ /opt/ppd/scripts/
COPY docs/ /opt/ppd/docs/
COPY deploy/ /opt/ppd/deploy/
COPY README.md HANDOFF_DCN9.md LICENSE /opt/ppd/

# A CPU-only code test; model/GPU/RDMA checks happen on the deployment hosts.
RUN python -B -m unittest discover -s scripts/tests -p 'test_*.py' -v
ENTRYPOINT ["python"]
CMD []

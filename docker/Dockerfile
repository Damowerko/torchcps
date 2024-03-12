FROM python:3.11-slim-bookworm

ARG USERNAME=default
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget git build-essential software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# install cuda
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && add-apt-repository contrib \
    && apt-get update \
    && apt-get -y install cuda-toolkit-12-4 \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

RUN groupadd --gid $USER_GID $USERNAME && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash

USER $USERNAME
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"
WORKDIR /home/$USERNAME/workspace

ENV VIRTUAL_ENV=/home/$USERNAME/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# upgrade pip
RUN pip install --upgrade pip

# install poetry
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && poetry config virtualenvs.create false

# install torchcps from github
RUN pip install "torch==2.1.*"

# install torchcps to preinstall dependencies
COPY poetry.lock pyproject.toml README.md /home/$USERNAME/torchcps/
RUN poetry install --directory=/home/$USERNAME/torchcps --no-root
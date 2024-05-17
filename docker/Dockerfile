FROM python:3.11-slim-bullseye

ARG USERNAME=default
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget git build-essential software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# install cuda
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && add-apt-repository contrib \
    && apt-get update \
    && apt-get -y install cuda-toolkit-12-2 \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=${CUDA_HOME}
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LIBRARY_PATH=${CUDA_HOME}/lib64
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64

RUN groupadd --gid $USER_GID $USERNAME && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash

USER $USERNAME
ENV HOME /home/$USERNAME
ENV USER $USERNAME
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"

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
COPY --chown=${USERNAME}:${USERNAME} poetry.lock pyproject.toml README.md /home/$USERNAME/torchcps/
RUN poetry install --directory=/home/$USERNAME/torchcps --no-root

# copy the rest of the repository
WORKDIR /home/$USERNAME/torchcps
RUN git config --global --add safe.directory /home/$USER/torchcps
COPY --chown=${USERNAME}:${USERNAME} . /home/$USERNAME/torchcps/
RUN poetry install --only-root
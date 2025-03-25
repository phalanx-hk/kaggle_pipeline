FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG UID
ARG GID
ARG PROJECT
ARG USERNAME=vscode

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=ja_JP.UTF-8 \
    TZ=Asia/Tokyo \
    apt_get_server=ftp.jaist.ac.jp/pub/Linux \
    MISE_INSTALL_PATH="/usr/local/bin/mise" \
    PATH="/mise/shims:$PATH" \
    MISE_DATA_DIR="/mise" \
    MISE_CONFIG_DIR="/mise" \
    MISE_CACHE_DIR="/mise/cache"


WORKDIR /workspace/projects/${PROJECT}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN \
    # apt
    --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    --mount=type=bind,source=.mise.toml,target=.mise.toml \
    rm /etc/apt/apt.conf.d/docker-clean \
    && sed -i s@archive.ubuntu.com@${apt_get_server}@g /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    git \
    locales \
    sudo \
    tmux \
    zsh \
    # setup locale
    && locale-gen ja_JP.UTF-8 \
    # install mise
    && curl https://mise.run | sh \
    # install cli tools
    && mise trust \
    && mise install \
    # add user
    && groupadd --gid ${GID} ${USERNAME} \
    && useradd -l --uid ${UID} --gid ${GID} -m ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/${USERNAME}

RUN --mount=type=bind,source=projects/${PROJECT}/uv.lock,target=uv.lock,readonly=false \
    --mount=type=bind,source=projects/${PROJECT}/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.mise.toml,target=.mise.toml \
    uv sync

ENV PATH="/workspace/projects/${PROJECT}/.venv/bin:$PATH"

USER ${USERNAME}

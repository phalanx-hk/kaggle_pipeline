FROM nvcr.io/nvidia/pytorch:24.02-py3

ARG UID=1000
ARG GID=1000
ARG USERNAME=vscode

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=ja_JP.UTF-8 \
    TZ=Asia/Tokyo \
    apt_get_server=ftp.jaist.ac.jp/pub/Linux \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/usr/

WORKDIR /workspace

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN \
    # apt
    --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    rm /etc/apt/apt.conf.d/docker-clean \
    && sed -i s@archive.ubuntu.com@${apt_get_server}@g /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    tmux \
    zsh \
    sudo \
    # install mise
    && curl https://mise.run | sh \
    # install cli tools
    && mise install \
    # add user
    && groupadd --gid ${GID} ${USERNAME} \
    && useradd -l --uid ${UID} --gid ${GID} -m ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/${USERNAME}

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync

USER ${USERNAME}

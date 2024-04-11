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

ENV  UV_VERSION=0.1.31

WORKDIR /workspace

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN \
    # apt
    --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    rm /etc/apt/apt.conf.d/docker-clean \
    && sed -i s@archive.ubuntu.com@${apt_get_server}@g /etc/apt/sources.list \
    && curl -sL https://raw.githubusercontent.com/eza-community/eza/main/deb.asc | gpg --dearmor -o /etc/apt/keyrings/gierens.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/gierens.gpg] http://deb.gierens.de stable main" | tee /etc/apt/sources.list.d/gierens.list \
    && chmod 644 /etc/apt/keyrings/gierens.gpg /etc/apt/sources.list.d/gierens.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    bat \
    fd-find \
    ripgrep \
    eza \
    shellcheck \
    tmux \
    zsh \
    sudo \
    && ln -s "$(which batcat)" /usr/local/bin/bat \
    && ln -s "$(which fdfind)" /usr/local/bin/fd \
    # just command runner
    && curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin \
    # hadolint
    && curl -fSL "https://github.com/hadolint/hadolint/releases/download/$(curl -s https://api.github.com/repos/hadolint/hadolint/releases/latest | jq -r '.tag_name')/hadolint-Linux-x86_64" -o /usr/local/bin/hadolint \
    && chmod +x /usr/local/bin/hadolint \
    # uv
    && pip install --no-cache-dir uv==${UV_VERSION} \
    # add user
    && groupadd --gid ${GID} ${USERNAME} \
    && useradd -l --uid ${UID} --gid ${GID} -m ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/${USERNAME}

RUN --mount=type=bind,source=requirements.txt,target=requirements.txt \
    --mount=type=bind,source=requirements-dev.txt,target=requirements-dev.txt \
    uv pip install -r requirements.txt \
    && uv pip install -r requirements-dev.txt

USER ${USERNAME}

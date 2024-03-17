#!/usr/bin/env bash

set -Eeuxo pipefail


function postCreateCommand() {
    pre-commit install
}

function main() {
    postCreateCommand
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main
fi

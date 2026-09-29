#!/usr/bin/env bash

api_image_tag() {
  local project_id="$1"
  local region="$2"
  local repo_root="$3"
  local commit
  commit="$(git -C "${repo_root}" rev-parse --short=12 HEAD)"
  printf '%s-docker.pkg.dev/%s/campaign-generator/api:%s\n' \
    "${region}" "${project_id}" "${commit}"
}

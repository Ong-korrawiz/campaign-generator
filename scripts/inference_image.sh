#!/usr/bin/env bash

MODEL_REVISION="989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

inference_image_tag() {
  local project_id="$1"
  local region="$2"
  printf '%s-docker.pkg.dev/%s/campaign-generator/baseline:%s\n' \
    "${region}" "${project_id}" "${MODEL_REVISION:0:12}"
}

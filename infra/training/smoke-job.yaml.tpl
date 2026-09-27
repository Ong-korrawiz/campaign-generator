serviceAccount: ${TRAINING_SERVICE_ACCOUNT}
scheduling:
  timeout: 1800s
workerPoolSpecs:
  - replicaCount: "1"
    machineSpec:
      machineType: e2-standard-4
    containerSpec:
      imageUri: gcr.io/google.com/cloudsdktool/google-cloud-cli:slim
      command: [bash, -c]
      args:
        - >-
          set -euo pipefail;
          gcloud storage cat "${INPUT_URI}";
          printf 'vertex-training-smoke-ok\n' |
          gcloud storage cp - "${OUTPUT_URI}/smoke-result.txt"

serviceAccount: ${TRAINING_SERVICE_ACCOUNT}
baseOutputDirectory:
  outputUriPrefix: ${OUTPUT_URI}
scheduling:
  strategy: SPOT
  timeout: 14400s
workerPoolSpecs:
  - replicaCount: "1"
    machineSpec:
      machineType: g2-standard-8
      acceleratorType: NVIDIA_L4
      acceleratorCount: 1
    containerSpec:
      imageUri: ${TRAINING_IMAGE_URI}
      args:
        - --dataset-uri=${DATASET_URI}
        - --output-uri=${OUTPUT_URI}
        - --run-id=${RUN_ID}

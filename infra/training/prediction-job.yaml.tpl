serviceAccount: ${TRAINING_SERVICE_ACCOUNT}
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
      command:
        - python
        - -m
        - campaign_generator.evaluation.predict_concepts
      args:
        - --test-file=${TEST_FILE_URI}
        - --output=${OUTPUT_URI}
        - --limit=36
        - --execute
${ADAPTER_ARG}

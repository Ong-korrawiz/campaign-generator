variable "project_id" {
  description = "Existing GCP project containing the platform resources."
  type        = string
}

variable "region" {
  description = "Cloud Run GPU region."
  type        = string
  default     = "asia-southeast1"
}

variable "inference_image" {
  description = "Immutable Artifact Registry image reference, including @sha256 digest."
  type        = string

  validation {
    condition     = can(regex("@sha256:[0-9a-f]{64}$", var.inference_image))
    error_message = "inference_image must end in @sha256:<64 lowercase hex characters>."
  }
}

variable "service_name" {
  description = "Private Cloud Run inference service name."
  type        = string
  default     = "campaign-baseline-inference"
}

variable "project_id" {
  description = "GCP project hosting the public API and private inference service."
  type        = string
}

variable "region" {
  description = "Cloud Run region."
  type        = string
  default     = "asia-southeast1"
}

variable "api_image" {
  description = "Immutable API image reference including a sha256 digest."
  type        = string

  validation {
    condition     = can(regex("@sha256:[0-9a-f]{64}$", var.api_image))
    error_message = "api_image must end in @sha256:<64 lowercase hex characters>."
  }
}

variable "inference_url" {
  description = "Base URL and OIDC audience of the private inference service."
  type        = string

  validation {
    condition     = can(regex("^https://", var.inference_url))
    error_message = "inference_url must be the HTTPS base URL of the private service."
  }
}

variable "api_service_name" {
  description = "Public API Cloud Run service name."
  type        = string
  default     = "campaign-api"
}

variable "inference_service_name" {
  description = "Private vLLM Cloud Run service name."
  type        = string
  default     = "campaign-baseline-inference"
}

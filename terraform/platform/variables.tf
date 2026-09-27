variable "project_id" {
  description = "Existing GCP project with billing enabled."
  type        = string
}

variable "region" {
  description = "Primary GCP region for storage, builds, training, and serving."
  type        = string
  default     = "asia-southeast1"
}

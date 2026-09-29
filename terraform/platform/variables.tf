variable "project_id" {
  description = "Existing GCP project with billing enabled."
  type        = string
}

variable "region" {
  description = "Primary GCP region for storage, builds, training, and serving."
  type        = string
  default     = "asia-southeast1"
}

variable "force_destroy_data_buckets" {
  description = "Delete all dataset and model artifact objects when destroying their buckets."
  type        = bool
  default     = false
}

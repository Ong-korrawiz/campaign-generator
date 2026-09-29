variable "project_id" {
  description = "Existing GCP project with billing enabled."
  type        = string
}

variable "region" {
  description = "Primary GCP region."
  type        = string
  default     = "asia-southeast1"
}

variable "force_destroy_state_bucket" {
  description = "Delete all Terraform state versions when destroying the state bucket."
  type        = bool
  default     = false
}

output "terraform_state_bucket" {
  description = "Bucket used by the platform and serving Terraform backends."
  value       = google_storage_bucket.terraform_state.name
}

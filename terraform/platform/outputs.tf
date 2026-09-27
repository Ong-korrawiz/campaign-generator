output "artifact_registry_repository" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.containers.repository_id}"
}

output "dataset_bucket" {
  value = google_storage_bucket.dataset.name
}

output "model_artifacts_bucket" {
  value = google_storage_bucket.model_artifacts.name
}

output "build_source_bucket" {
  value = google_storage_bucket.build_source.name
}

output "service_accounts" {
  value = {
    for name, account in google_service_account.runtime : name => account.email
  }
}

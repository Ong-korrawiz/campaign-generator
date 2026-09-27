output "inference_service_name" {
  value = google_cloud_run_v2_service.inference.name
}

output "inference_url" {
  value = google_cloud_run_v2_service.inference.uri
}

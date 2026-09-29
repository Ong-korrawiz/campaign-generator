locals {
  api_service_account = "campaign-api@${var.project_id}.iam.gserviceaccount.com"
}

resource "google_cloud_run_v2_service" "api" {
  project              = var.project_id
  name                 = var.api_service_name
  location             = var.region
  deletion_protection  = false
  ingress              = "INGRESS_TRAFFIC_ALL"
  invoker_iam_disabled = true

  template {
    service_account                  = local.api_service_account
    timeout                          = "900s"
    max_instance_request_concurrency = 1

    scaling {
      min_instance_count = 0
      max_instance_count = 1
    }

    containers {
      image = var.api_image

      ports {
        name           = "http1"
        container_port = 8080
      }

      env {
        name  = "INFERENCE_URL"
        value = var.inference_url
      }
      env {
        name  = "INFERENCE_AUDIENCE"
        value = var.inference_url
      }
      env {
        name  = "FEEDBACK_BUCKET"
        value = "${var.project_id}-feedback"
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
        cpu_idle = true
      }

      startup_probe {
        initial_delay_seconds = 0
        period_seconds        = 10
        timeout_seconds       = 3
        failure_threshold     = 3
        http_get {
          path = "/health"
          port = 8080
        }
      }
    }
  }
}

locals {
  api_service_account       = "campaign-api@${var.project_id}.iam.gserviceaccount.com"
  inference_service_account = "campaign-inference@${var.project_id}.iam.gserviceaccount.com"
}

resource "google_cloud_run_v2_service" "inference" {
  project             = var.project_id
  name                = var.service_name
  location            = var.region
  deletion_protection = false
  ingress             = "INGRESS_TRAFFIC_ALL"

  template {
    service_account                  = local.inference_service_account
    timeout                          = "600s"
    max_instance_request_concurrency = 4
    gpu_zonal_redundancy_disabled    = true

    scaling {
      min_instance_count = 0
      max_instance_count = 1
    }

    containers {
      image = var.inference_image

      ports {
        name           = "http1"
        container_port = 8080
      }

      resources {
        limits = {
          cpu              = "4"
          memory           = "16Gi"
          "nvidia.com/gpu" = "1"
        }
        cpu_idle          = false
        startup_cpu_boost = true
      }

      startup_probe {
        failure_threshold     = 1200
        initial_delay_seconds = 0
        period_seconds        = 1
        timeout_seconds       = 1

        tcp_socket {
          port = 8080
        }
      }
    }

    node_selector {
      accelerator = "nvidia-l4"
    }
  }
}

resource "google_cloud_run_v2_service_iam_member" "api_invoker" {
  project  = var.project_id
  location = google_cloud_run_v2_service.inference.location
  name     = google_cloud_run_v2_service.inference.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${local.api_service_account}"
}

locals {
  services = toset(["compute.googleapis.com", "iam.googleapis.com", "iap.googleapis.com", "sqladmin.googleapis.com",
  "servicenetworking.googleapis.com", "secretmanager.googleapis.com", "artifactregistry.googleapis.com"])
  registry     = "${var.region}-docker.pkg.dev/${var.project_id}/${var.name}"
  sql_username = trimsuffix(google_service_account.runtime.email, ".gserviceaccount.com")
  runtime_config = {
    project_id      = var.project_id
    registry        = local.registry
    backend_image   = var.backend_image
    worker_image    = var.gpu_enabled ? var.gpu_backend_image : var.backend_image
    web_image       = var.web_image
    proxy_image     = var.proxy_image
    build_commit    = var.build_commit
    gpu_enabled     = var.gpu_enabled
    runtime_enabled = var.runtime_enabled
    licence_policy  = "commercial"
    sql_connection  = google_sql_database_instance.main.connection_name
    sql_username    = local.sql_username
    secret_resource = "${google_secret_manager_secret.runtime.id}/versions/${var.secret_version}"
  }
}
resource "google_project_service" "required" {
  for_each           = local.services
  service            = each.value
  disable_on_destroy = false
}
resource "google_compute_network" "main" {
  name                    = var.name
  auto_create_subnetworks = false
  depends_on              = [google_project_service.required]
}
resource "google_compute_subnetwork" "main" {
  name                     = var.name
  ip_cidr_range            = "10.70.0.0/24"
  region                   = var.region
  network                  = google_compute_network.main.id
  private_ip_google_access = true
}
resource "google_compute_router" "main" {
  name    = var.name
  region  = var.region
  network = google_compute_network.main.id
}
# NAT is for host package/image provisioning and the SQL proxy. Application containers have only an internal network.
resource "google_compute_router_nat" "main" {
  name                               = var.name
  router                             = google_compute_router.main.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "LIST_OF_SUBNETWORKS"
  subnetwork {
    name                    = google_compute_subnetwork.main.id
    source_ip_ranges_to_nat = ["ALL_IP_RANGES"]
  }
  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}
resource "google_service_account" "runtime" {
  account_id   = "${var.name}-runtime"
  display_name = "Veritas single-host runtime"
  depends_on   = [google_project_service.required]
}
resource "google_compute_firewall" "iap" {
  name                    = "${var.name}-iap"
  network                 = google_compute_network.main.name
  source_ranges           = ["35.235.240.0/20"]
  target_service_accounts = [google_service_account.runtime.email]
  allow {
    protocol = "tcp"
    ports    = ["22", "8080"]
  }
  log_config { metadata = "EXCLUDE_ALL_METADATA" }
}
resource "google_compute_global_address" "sql" {
  name          = "${var.name}-sql"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.main.id
}
resource "google_service_networking_connection" "sql" {
  network                 = google_compute_network.main.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.sql.name]
  deletion_policy         = "ABANDON"
}
resource "google_sql_database_instance" "main" {
  name                = "${var.name}-postgres"
  region              = var.region
  database_version    = "POSTGRES_16"
  deletion_protection = true
  depends_on          = [google_service_networking_connection.sql]
  settings {
    tier                        = "db-custom-2-7680"
    edition                     = "ENTERPRISE"
    availability_type           = "REGIONAL"
    disk_type                   = "PD_SSD"
    disk_size                   = 50
    disk_autoresize             = true
    deletion_protection_enabled = true
    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.main.id
      ssl_mode        = "ENCRYPTED_ONLY"
    }
    database_flags {
      name  = "cloudsql.iam_authentication"
      value = "on"
    }
    backup_configuration {
      enabled                        = true
      point_in_time_recovery_enabled = true
      start_time                     = "03:00"
      transaction_log_retention_days = 7
      backup_retention_settings { retained_backups = 14 }
    }
    maintenance_window {
      day          = 7
      hour         = 5
      update_track = "stable"
    }
  }
  lifecycle { prevent_destroy = true }
}
resource "google_sql_database" "app" {
  name     = "veritas"
  instance = google_sql_database_instance.main.name
  lifecycle { prevent_destroy = true }
}
resource "google_sql_user" "runtime" {
  name     = local.sql_username
  instance = google_sql_database_instance.main.name
  type     = "CLOUD_IAM_SERVICE_ACCOUNT"
}
resource "google_project_iam_member" "sql" {
  for_each = toset(["roles/cloudsql.client", "roles/cloudsql.instanceUser"])
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.runtime.email}"
  condition {
    title      = "veritas-instance-only"
    expression = "resource.name == 'projects/${var.project_id}/instances/${google_sql_database_instance.main.name}'"
  }
}
resource "google_artifact_registry_repository" "images" {
  location      = var.region
  repository_id = var.name
  format        = "DOCKER"
  depends_on    = [google_project_service.required]
  lifecycle { prevent_destroy = true }
}
resource "google_artifact_registry_repository_iam_member" "pull" {
  location   = var.region
  repository = google_artifact_registry_repository.images.name
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.runtime.email}"
}
# Only the secret container is managed here. Values are added separately, never passed through Terraform/state/metadata.
resource "google_secret_manager_secret" "runtime" {
  secret_id = "${var.name}-runtime"
  replication {
    auto {}
  }
  depends_on = [google_project_service.required]
  lifecycle { prevent_destroy = true }
}
resource "google_secret_manager_secret_iam_member" "runtime" {
  secret_id = google_secret_manager_secret.runtime.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}
resource "google_compute_disk" "persistent" {
  for_each = { data = var.data_disk_gb, models = var.models_disk_gb }
  name     = "${var.name}-${each.key}"
  zone     = var.zone
  type     = "pd-balanced"
  size     = each.value
  lifecycle { prevent_destroy = true }
}
# Scheduled disks are crash-consistent; use the documented quiesced SQL+disk checkpoint before upgrades/restores.
resource "google_compute_resource_policy" "snapshots" {
  name   = "${var.name}-daily"
  region = var.region
  snapshot_schedule_policy {
    schedule {
      daily_schedule {
        days_in_cycle = 1
        start_time    = "04:00"
      }
    }
    retention_policy {
      max_retention_days    = 14
      on_source_disk_delete = "KEEP_AUTO_SNAPSHOTS"
    }
    snapshot_properties { storage_locations = [var.region] }
  }
}
resource "google_compute_disk_resource_policy_attachment" "snapshots" {
  for_each = google_compute_disk.persistent
  name     = google_compute_resource_policy.snapshots.name
  disk     = each.value.name
  zone     = var.zone
}
resource "google_compute_instance" "app" {
  name                      = var.name
  zone                      = var.zone
  machine_type              = var.gpu_enabled ? "g2-standard-16" : "n2-standard-16"
  allow_stopping_for_update = true
  deletion_protection       = true
  boot_disk {
    initialize_params {
      image = var.boot_image
      size  = 100
      type  = "pd-balanced"
    }
  }
  dynamic "attached_disk" {
    for_each = google_compute_disk.persistent
    content {
      source      = attached_disk.value.id
      device_name = "veritas-${attached_disk.key}"
    }
  }
  network_interface { subnetwork = google_compute_subnetwork.main.id } # deliberately no access_config/public IP
  service_account {
    email  = google_service_account.runtime.email
    scopes = ["cloud-platform"]
  }
  scheduling {
    on_host_maintenance = var.gpu_enabled ? "TERMINATE" : "MIGRATE"
    automatic_restart   = true
    provisioning_model  = "STANDARD"
  }
  dynamic "guest_accelerator" {
    for_each = var.gpu_enabled ? [1] : []
    content {
      type  = "nvidia-l4"
      count = 1
    }
  }
  shielded_instance_config {
    enable_secure_boot          = !var.gpu_enabled
    enable_vtpm                 = true
    enable_integrity_monitoring = true
  }
  metadata = {
    enable-oslogin           = "TRUE"
    block-project-ssh-keys   = "TRUE"
    disable-legacy-endpoints = "TRUE"
    startup-script = templatefile("${path.module}/startup.sh.tftpl", {
      config = base64encode(jsonencode(local.runtime_config))
      assets = { for name in ["runtime.py", "entrypoint.py", "preflight.py", "release.py", "db_admin.py"] : name => filebase64("${path.module}/${name}") }
    })
  }
  depends_on = [google_compute_router_nat.main, google_project_iam_member.sql,
  google_secret_manager_secret_iam_member.runtime, google_artifact_registry_repository_iam_member.pull]
  lifecycle {
    precondition {
      condition     = !var.gpu_enabled || var.gpu_backend_image != null
      error_message = "The GPU variant requires a same-commit CUDA backend image."
    }
    precondition {
      condition = alltrue([for image in compact([var.backend_image, var.web_image, var.proxy_image, var.gpu_backend_image]) :
      startswith(image, "${local.registry}/")])
      error_message = "Mirror all runtime images into this module's repository; VM pull permission is scoped to it."
    }
  }
}
resource "google_iap_tunnel_instance_iam_member" "web" {
  for_each = var.iap_members
  project  = var.project_id
  zone     = var.zone
  instance = google_compute_instance.app.name
  role     = "roles/iap.tunnelResourceAccessor"
  member   = each.value
  condition {
    title      = "web-only"
    expression = "destination.port == 8080"
  }
}
resource "google_iap_tunnel_instance_iam_member" "operator" {
  for_each = var.operator_members
  project  = var.project_id
  zone     = var.zone
  instance = google_compute_instance.app.name
  role     = "roles/iap.tunnelResourceAccessor"
  member   = each.value
  condition {
    title      = "ssh-only"
    expression = "destination.port == 22"
  }
}
resource "google_compute_instance_iam_member" "operator" {
  for_each      = var.operator_members
  project       = var.project_id
  zone          = var.zone
  instance_name = google_compute_instance.app.name
  role          = "roles/compute.osAdminLogin"
  member        = each.value
}
resource "google_service_account_iam_member" "operator" {
  for_each           = var.operator_members
  service_account_id = google_service_account.runtime.name
  role               = "roles/iam.serviceAccountUser"
  member             = each.value
}
# gcloud tunnel discovery needs only instance get/list; no project-wide instance-admin role.
resource "google_project_iam_custom_role" "tunnel_discovery" {
  role_id     = "${replace(var.name, "-", "_")}_tunnel_discovery"
  title       = "Veritas tunnel discovery"
  permissions = ["compute.instances.get", "compute.instances.list", "compute.projects.get"]
}
resource "google_project_iam_member" "tunnel_discovery" {
  for_each = setunion(var.iap_members, var.operator_members)
  project  = var.project_id
  role     = google_project_iam_custom_role.tunnel_discovery.name
  member   = each.value
}

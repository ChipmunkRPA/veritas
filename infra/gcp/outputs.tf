output "image_repository" { value = local.registry }
output "runtime_secret" { value = google_secret_manager_secret.runtime.id }
output "sql_connection" { value = google_sql_database_instance.main.connection_name }
output "sql_iam_username" { value = local.sql_username }
output "web_tunnel" {
  value = "gcloud compute start-iap-tunnel ${var.name} 8080 --local-host-port=localhost:8080 --zone=${var.zone} --project=${var.project_id}"
}
output "operator_ssh" {
  value = "gcloud compute ssh ${var.name} --tunnel-through-iap --zone=${var.zone} --project=${var.project_id}"
}

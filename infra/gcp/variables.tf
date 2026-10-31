variable "project_id" {
  type        = string
  description = "Existing, billing-enabled deployment project; no credentials belong in tfvars."
}
variable "name" {
  type    = string
  default = "veritas"
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,19}$", var.name))
    error_message = "Use 3–20 lower-case letters, digits or hyphens, starting with a letter."
  }
}
variable "region" {
  type    = string
  default = "us-central1"
}
variable "zone" {
  type    = string
  default = "us-central1-a"
  validation {
    condition     = startswith(var.zone, "${var.region}-")
    error_message = "The VM zone must be in the Cloud SQL/disk region."
  }
}
variable "backend_image" {
  type        = string
  description = "CPU backend image in this module's Artifact Registry, pinned by sha256."
  validation {
    condition     = can(regex("@sha256:[0-9a-f]{64}$", var.backend_image))
    error_message = "A backend image digest is required, not a mutable tag."
  }
}
variable "web_image" {
  type        = string
  description = "Frontend image in this module's Artifact Registry, pinned by sha256."
  validation {
    condition     = can(regex("@sha256:[0-9a-f]{64}$", var.web_image))
    error_message = "A web image digest is required."
  }
}
variable "proxy_image" {
  type        = string
  description = "Reviewed Cloud SQL Auth Proxy v2 image, mirrored to this module's Artifact Registry and pinned by sha256."
  validation {
    condition     = can(regex("@sha256:[0-9a-f]{64}$", var.proxy_image))
    error_message = "A reviewed Cloud SQL Auth Proxy v2 image digest is required."
  }
}
variable "build_commit" {
  type        = string
  description = "Full source commit baked into both backend images and runtime-release.json."
  validation {
    condition     = can(regex("^[0-9a-f]{40}$", var.build_commit))
    error_message = "Use the complete 40-character source commit."
  }
}
variable "boot_image" {
  type        = string
  description = "Specific reviewed Ubuntu 24.04 amd64 image reference; GPU variant must have a working NVIDIA driver >=560 preinstalled."
  validation {
    condition     = can(regex("^(https://www\\.googleapis\\.com/compute/v1/)?projects/[a-z0-9][a-z0-9-]*/global/images/[a-z]([-a-z0-9]{0,61}[a-z0-9])?$", var.boot_image))
    error_message = "Use projects/PROJECT/global/images/IMAGE or its https://www.googleapis.com/compute/v1/ self-link. Image families, abbreviated aliases, queries and fragments are not allowed."
  }
}
variable "gpu_enabled" {
  type    = bool
  default = false
}
variable "gpu_backend_image" {
  type        = string
  default     = null
  nullable    = true
  description = "CUDA backend built from the same commit; required for the single-L4 GPU variant."
  validation {
    condition     = var.gpu_backend_image == null ? true : can(regex("@sha256:[0-9a-f]{64}$", var.gpu_backend_image))
    error_message = "The CUDA image must be pinned by sha256."
  }
}
variable "runtime_enabled" {
  type        = bool
  default     = false
  description = "Keep false until image publication, model/evidence-volume seeding, secrets, and database migration are complete."
}
variable "secret_version" {
  type        = string
  default     = "1"
  description = "Explicit numeric version of the runtime secret, avoiding rotation surprises at reboot."
  validation {
    condition     = can(regex("^[1-9][0-9]*$", var.secret_version))
    error_message = "Pin a numeric Secret Manager version."
  }
}
variable "iap_members" {
  type        = set(string)
  default     = []
  description = "user: or group: principals allowed to tunnel to the UI on port 8080; app sign-in is still required."
  validation {
    condition     = alltrue([for m in var.iap_members : can(regex("^(user|group):[^ ]+@[^ ]+$", m))])
    error_message = "Only explicit user or group email principals are supported."
  }
}
variable "operator_members" {
  type        = set(string)
  default     = []
  description = "Privileged operators: IAP SSH, OS Login administrator and use of the VM service account."
  validation {
    condition     = alltrue([for m in var.operator_members : can(regex("^(user|group):[^ ]+@[^ ]+$", m))])
    error_message = "Only explicit user or group email principals are supported."
  }
}
variable "data_disk_gb" {
  type    = number
  default = 500
  validation {
    condition     = var.data_disk_gb >= 100
    error_message = "Reserve at least 100 GB for uploads, evidence, reports and transient decoding."
  }
}
variable "models_disk_gb" {
  type    = number
  default = 300
  validation {
    condition     = var.models_disk_gb >= 100
    error_message = "Reserve at least 100 GB for pinned model caches."
  }
}

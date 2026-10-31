# Google Cloud deployment readiness

The executable deployment bundle and operator runbook are in [`infra/gcp`](../infra/gcp/README.md).

The supported initial topology is one private Compute Engine VM with separate API and worker containers, durable ext4 evidence
and model disks, and private Cloud SQL PostgreSQL. A one-L4 NVIDIA worker variant is included. Access uses IAP TCP forwarding plus
Veritas application sign-in and case RBAC. The runtime defaults to disabled until assets, secrets and database grants are provisioned.

Application startup now waits for both API/database health and worker readiness before starting web. Worker readiness requires a successful device operation, live worker threads and recent queue communication; it does not certify model warmup or detection accuracy. Startup waits are bounded and failures clean up the partial stack. These dependencies gate startup only; later unhealthy containers require monitoring and operator recovery.

This is structural readiness, not a completed deployment or an accuracy improvement. The bundle documents the remaining image,
live-GCP, GPU parity, performance, security and coordinated-restore acceptance checks. No Google Cloud resources have been created.

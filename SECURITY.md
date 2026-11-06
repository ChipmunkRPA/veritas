# Security reporting

Use GitHub private vulnerability reporting when available. Otherwise contact
sales@chipmunkrpa.com with a short description and a way to follow up privately.
Do not post an exploit with working credentials or sensitive media in a public issue.

Include the affected revision, configuration category, impact, and minimal reproduction
using synthetic data. Do not send API keys, Stripe secrets, session tokens, personal
uploads, or private evaluation datasets. There is no guaranteed response-time commitment.

## Deployment boundaries

This is an actively developed screening system, not a certified authenticity service.
Run untrusted media with resource limits and keep dependencies current. Production
deployment requires authentication, protected transport, appropriate retention controls,
and a review of the known limitations and applicable dependency licenses.

The repository intentionally excludes runtime databases, upload stores, signing keys,
credentials, generated customer reports, and model weights. Local configuration belongs
in ignored environment files. The example environment file contains no usable credentials.

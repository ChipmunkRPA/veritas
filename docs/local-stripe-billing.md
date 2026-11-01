# Local Stripe billing

Chipmunk Verify charges **US$9.99 for one automated analysis of one source**. This is a one-time payment, not a subscription. An inconclusive assessment is a possible completed result. Expert analysis and expert witness work are separately contracted through **sales@chipmunkrpa.com**.

## Current integration

- Stripe-hosted Checkout handles card details. The application does not collect or store card numbers.
- A signed-in account owns the upload and payment order. Anonymous local administrator access is disabled whenever billing is enabled.
- The server fixes the currency to `usd`, the unit amount to `999`, and the quantity to one. Browser-supplied prices are not accepted.
- Files are validated and saved before checkout. An analysis job is created only after a signed webhook and a fresh Stripe API response verify the payment, session, order, amount, currency, quantity, and payment environment.
- Repeated upload requests, checkout retries, and duplicate webhook delivery reuse the saved order. Fulfillment creates one analysis and one job transactionally.
- Returning from Checkout does not prove payment. The return page polls the order's verified status.
- Payment methods follow the Stripe account's Dashboard settings. A delayed payment remains pending until Stripe confirms it; failed payments do not start an analysis.
- Direct legacy, batch, demo, reanalysis, rerun, and refusion submission routes cannot bypass checkout while billing is enabled. Paid batch checkout is not implemented; submit one source at a time.
- Before model execution, the worker independently checks the committed payment receipt against the exact analysis, owner, source, profile, US$9.99 amount, and payment environment. Old unpaid queued jobs, unpaid recovered jobs, and stale batch expansions cannot start. A test payment cannot authorize work in live mode. Completed historical results and their exports remain accessible.
- Internal worker retries do not charge again. If an analysis fails technically, retain the order ID and contact support. Automatic refunds and a customer self-service retry of a failed run are not implemented; the UI must not promise either or ask the customer to pay again to resolve that failure.

## Sandbox connection

The local CLI profile is `chipmunk-verify`. Stripe sandbox authorization is separate from enabling live payments. No live payments should be used for local verification.

From the repository root:

```sh
stripe login --project-name chipmunk-verify
backend/.venv/bin/python backend/scripts/stripe_local.py configure
backend/.venv/bin/python backend/scripts/stripe_local.py listen
```

`configure` verifies the authorized sandbox account and writes the test key, local webhook secret, test mode, and local browser origin into the ignored `backend/.env` file with owner-only permissions. It refuses to replace an existing live configuration. The helper never prints credentials. Restart the API after configuring it.

`listen` keeps forwarding test Checkout events to `http://127.0.0.1:8160/api/v2/billing/webhook`. Keep this process running during local checkout testing. It checks that the saved secret still matches and redacts the secret from CLI output. A Dashboard webhook secret and a CLI forwarding secret are different credentials.

Settings:

| Setting | Local value |
| --- | --- |
| `VERITAS_BILLING_MODE` | `test` (`disabled` restores the existing local free workflow) |
| `VERITAS_STRIPE_SECRET_KEY` | Authorized sandbox secret, never committed |
| `VERITAS_STRIPE_WEBHOOK_SECRET` | CLI forwarding secret, never committed |
| `VERITAS_BILLING_APP_ORIGIN` | `http://127.0.0.1:5230` |

Use the account-management CLI to create named users; public registration is not implemented. A CLI-generated credential should be handled privately and changed by its owner before public deployment.

## Local verification performed

On October 31, 2026, the authorized Chipmunk Robotics sandbox completed a real Stripe-hosted test Checkout for **999 USD cents** using Stripe's test card. Stripe delivered `checkout.session.completed` to the local webhook, which returned HTTP 200. The document analysis finished successfully in approximately 14 seconds. Two recorded duplicate notification replays returned HTTP 200 and left exactly one analysis and one analysis job. All 80 pre-existing analysis rows were unchanged.

The verification record is in `backend/data/billing-validation/sandbox-result.json`. No live card was charged. The browser is signed into the generated `chipmunk-local` account; its generated local password is stored in the ignored, owner-readable `backend/data/billing-validation/local-account.txt` file, not this document.

The running API also rejected a damaged PNG with HTTP 422 without creating a payment order, rejected the old unpaid text submission route with HTTP 402, and rejected anonymous access with HTTP 401. The final focused backend suite passed 40 tests across billing, prepayment decoding, and secure local setup. The frontend's focused billing tests and production build passed.

Run the focused tests without inheriting the local paid-mode setting:

```sh
VERITAS_BILLING_MODE=disabled backend/.venv/bin/python -m pytest -q \
  backend/tests/test_stripe_billing.py backend/tests/test_billing_preflight.py \
  backend/tests/test_stripe_local_setup.py backend/tests/test_billing_worker_gate.py
```

The billing tests explicitly enable an isolated test configuration with mocked Stripe responses. The separate hosted-checkout verification above used the actual Stripe sandbox.

Payment-first regression checks also cover forged success query parameters, delayed or failed payments, a Stripe verification outage, and unpaid work already in the queue. The upload screen states that payment must be confirmed before analysis; the checkout path never calls a free analysis submission endpoint.

The Stripe account's existing hosted-checkout business name is **Chipmunk Technologies**; the purchased item is **Chipmunk Verify — one analysis**. This test did not change the account's business identity or live payment settings.

## Before GCP deployment

Set an HTTPS application origin, configure a persistent Stripe webhook endpoint, store secrets in the deployment secret store, and verify account access and payment behavior in the deployed sandbox. Live mode requires matching live credentials and webhook configuration. Public account registration, refund policy and automation, tax treatment, abuse controls for unpaid uploads, and payment reconciliation after prolonged webhook outages need a production review. The local implementation does not resolve these business and deployment decisions automatically.

## References

The implementation follows Stripe's guidance on [webhook fulfillment](https://docs.stripe.com/checkout/fulfillment), [idempotent API requests](https://docs.stripe.com/api/idempotent_requests), and [webhook signature verification](https://docs.stripe.com/events/manage-webhook-endpoints#signature-errors).

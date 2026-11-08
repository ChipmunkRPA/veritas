# Verification and publication

The project owner directed on 8 November 2026 that updates be tested locally
and pushed directly to GitHub, without running GitHub Actions workflows.

- Do not create, enable, dispatch or rerun GitHub Actions workflows unless the
  user explicitly changes this instruction.
- Do not re-enable Actions as part of an automatic goal continuation,
  independent-testing task or publication step. Actions is disabled for
  `ChipmunkRPA/veritas`.
- Run appropriate checks locally, inspect their actual results, then commit and
  push the reviewed changes directly.
- Keep historical workflow definitions, failures and receipts as historical
  evidence. Do not describe them as newly executed verification.
- Independent evaluation may continue locally or through separately authorized
  reviewers; it does not require a GitHub workflow run.

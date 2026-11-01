#!/usr/bin/env python3
"""Configure or forward Stripe sandbox payments without printing credentials.

Run with the backend virtual environment after `stripe login --project-name
chipmunk-verify`. This helper deliberately cannot enable live billing.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import httpx
from dotenv import dotenv_values, set_key

BACKEND = Path(__file__).resolve().parents[1]
ENV_FILE = BACKEND / ".env"
PROFILE = "chipmunk-verify"
EVENTS = "checkout.session.completed,checkout.session.async_payment_succeeded,checkout.session.async_payment_failed,checkout.session.expired"
SECRET = re.compile(r"(?:[srp]k_(?:test|live)_|whsec_)[A-Za-z0-9_]+")


def cli() -> str:
    executable = shutil.which("stripe")
    if not executable:
        raise RuntimeError("Install the official Stripe CLI first.")
    return executable


def signing_secret() -> str:
    result = subprocess.run(
        [cli(), "listen", "--project-name", PROFILE, "--print-secret", "--skip-update"],
        capture_output=True, text=True, timeout=40, check=False,
    )
    match = re.search(r"whsec_[A-Za-z0-9]+", result.stdout)
    if result.returncode or not match:
        raise RuntimeError("Stripe could not supply a test webhook secret. Sign in again with the chipmunk-verify profile.")
    return match.group()


def configure() -> None:
    existing = dotenv_values(ENV_FILE)
    if existing.get("VERITAS_BILLING_MODE") == "live":
        raise RuntimeError("This sandbox helper refuses to replace a live billing configuration.")
    config = Path.home() / ".config/stripe/config.toml"
    with config.open("rb") as stream:
        profile = tomllib.load(stream).get(PROFILE, {})
    key = profile.get("test_mode_api_key", "")
    if not key.startswith(("sk_test_", "rk_test_")):
        raise RuntimeError("Run stripe login --project-name chipmunk-verify and authorize your sandbox account first.")
    response = httpx.get("https://api.stripe.com/v1/account", auth=(key, ""), timeout=25)
    if response.status_code != 200:
        raise RuntimeError(f"Stripe rejected the saved test connection (HTTP {response.status_code}); sign in again.")
    account = response.json()
    secret = signing_secret()
    ENV_FILE.touch(mode=0o600, exist_ok=True)
    os.chmod(ENV_FILE, 0o600)
    for name, value in {
        "VERITAS_BILLING_MODE": "test",
        "VERITAS_STRIPE_SECRET_KEY": key,
        "VERITAS_STRIPE_WEBHOOK_SECRET": secret,
        "VERITAS_BILLING_APP_ORIGIN": "http://127.0.0.1:5230",
    }.items():
        set_key(ENV_FILE, name, value)
    os.chmod(ENV_FILE, 0o600)
    print(f"Stripe sandbox account {account.get('id')} configured for US$9.99 per analysis.")
    print("Credentials saved only in the ignored backend/.env file. Restart the API, then run this helper with listen.")


def listen() -> int:
    values = dotenv_values(ENV_FILE)
    if values.get("VERITAS_BILLING_MODE") != "test":
        raise RuntimeError("Configure sandbox billing first. This helper forwards test events only.")
    if signing_secret() != values.get("VERITAS_STRIPE_WEBHOOK_SECRET"):
        raise RuntimeError("The webhook secret changed. Run configure again and restart the API before listening.")
    command = [cli(), "listen", "--project-name", PROFILE, "--skip-update", "--color", "off",
               "--events", EVENTS, "--forward-to", "http://127.0.0.1:8160/api/v2/billing/webhook"]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(SECRET.sub("[redacted]", line), end="", flush=True)
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            return 130


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("configure", "listen"))
    args = parser.parse_args()
    try:
        if args.command == "configure":
            configure()
            return 0
        return listen()
    except (RuntimeError, OSError, httpx.HTTPError, subprocess.TimeoutExpired, ValueError) as error:
        # No provider response bodies or credential-bearing subprocess output.
        message = str(error) if isinstance(error, RuntimeError) else type(error).__name__
        print(f"Stripe local setup: {SECRET.sub('[redacted]', message)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

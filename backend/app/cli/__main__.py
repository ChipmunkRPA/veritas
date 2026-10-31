import sys

from app.cli import main
from app.core.logging import configure_logging

# N-13: the command line logs as the services do (JSON in prod), on standard error: what a command prints is its answer
configure_logging(sys.stderr)
raise SystemExit(main())

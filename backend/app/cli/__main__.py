import sys

from app.cli import main

# Offline package verification needs no service settings or application logging.
if sys.argv[1:2] != ["verify"]:
    from app.core.logging import configure_logging

    configure_logging(sys.stderr)
raise SystemExit(main())

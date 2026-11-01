#!/usr/bin/env bash
# Manage only processes started by this local development launcher.
set -eu

backend_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
state_dir=${VERITAS_DEV_STATE_DIR:-"$backend_dir/data/jobs"}
action=${1:-}
name=${2:-}
case "$name" in dev-api|dev-web) ;; *) echo "Unknown development service: $name" >&2; exit 2 ;; esac
mkdir -p "$state_dir"
pidfile="$state_dir/$name.pid"
logfile="$state_dir/$name.log"
lock="$state_dir/$name.lock"
if ! mkdir "$lock" 2>/dev/null; then
  echo "$name is being managed by another command ($lock exists)" >&2
  exit 1
fi
trap 'rmdir "$lock"' EXIT

identity() { ps -p "$1" -o lstart= -o command= 2>/dev/null || true; }
read_record() {
  recorded_pid=''
  recorded_identity=''
  if [ -f "$pidfile" ]; then
    { IFS= read -r recorded_pid || true; IFS= read -r recorded_identity || true; } < "$pidfile"
  fi
  case "$recorded_pid" in ''|*[!0-9]*) recorded_pid='';; esac
}
is_managed_process() {
  [ -n "$recorded_pid" ] && [ -n "$recorded_identity" ] &&
    [ "$(identity "$recorded_pid")" = "$recorded_identity" ]
}

case "$action" in
  start)
    shift 2
    [ "${1:-}" = -- ] && shift
    [ "$#" -gt 0 ] || { echo "No command supplied" >&2; exit 2; }
    read_record
    if is_managed_process; then echo "$name is already running (PID $recorded_pid)"; exit 0; fi
    if [ -f "$pidfile" ]; then echo "Removing stale $pidfile (no process was stopped)"; rm -f "$pidfile"; fi
    nohup "$@" > "$logfile" 2>&1 < /dev/null &
    pid=$!
    sleep 1
    process_identity=$(identity "$pid")
    if [ -z "$process_identity" ] || ! kill -0 "$pid" 2>/dev/null; then
      echo "$name exited during startup; see $logfile" >&2
      exit 1
    fi
    printf '%s\n%s\n' "$pid" "$process_identity" > "$pidfile"
    echo "$name started (PID $pid; log: $logfile)"
    ;;
  stop)
    read_record
    if [ ! -f "$pidfile" ]; then echo "$name is not managed by this launcher"; exit 0; fi
    if ! is_managed_process; then
      echo "Removing stale $pidfile; no process was stopped"
      rm -f "$pidfile"
      exit 0
    fi
    kill "$recorded_pid"
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      if ! is_managed_process; then rm -f "$pidfile"; echo "$name stopped"; exit 0; fi
      sleep 1
    done
    echo "$name has not stopped; PID record retained in $pidfile" >&2
    exit 1
    ;;
  *) echo "Usage: $0 {start|stop} {dev-api|dev-web} [-- command ...]" >&2; exit 2 ;;
esac

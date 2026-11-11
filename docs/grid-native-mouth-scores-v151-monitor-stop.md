# Locked scoring paused by local process monitoring

The first guarded scoring stage completed all eleven matched cases. On the
next stage, the process monitor encountered a permission error and stopped
the child before it produced a score receipt. The [observed stop receipt](evaluation-panels/grid-native-mouth-scores-monitor-stop-v151.json)
binds the resource profile, samples and worker log. Available memory at the
observed admission was 7.50 GiB; this was a monitoring failure.

The completed first stage and the failed second attempt are retained. A
separate execution context must complete the remaining cases under the
same resource guard, inputs, heads, settings and scoring arithmetic. The
first stage remains part of the final comparison, and the failed attempt
contributes no detector outcome. App defaults did not change.

"""
spawn_proxy_jobs.py -- submit generate() calls against the DEPLOYED
cd-rethink-chair app asynchronously (spawn, not remote/run). Each call only
needs the local connection alive for the few seconds it takes to submit the
job; the remote execution then continues fully independently on Modal's
infrastructure, decoupled from this process -- unlike `modal run --detach`,
which has shown it can still be cancelled if the local client's connection
drops mid-stream.

Usage:
    python spawn_proxy_jobs.py
"""
import modal

f = modal.Function.from_name("cd-rethink-chair", "generate")

jobs = [
    dict(mode="proxy", stats_source="vcd", seed=0, out_name="proxy_vcdstats/seed0.jsonl"),
    dict(mode="proxy", stats_source="vcd", seed=1, out_name="proxy_vcdstats/seed1.jsonl"),
    dict(mode="proxy", stats_source="vcd", seed=2, out_name="proxy_vcdstats/seed2.jsonl"),
    dict(mode="proxy", stats_source="sid", seed=0, out_name="proxy_sidstats/seed0.jsonl"),
    dict(mode="proxy", stats_source="sid", seed=1, out_name="proxy_sidstats/seed1.jsonl"),
    dict(mode="proxy", stats_source="sid", seed=2, out_name="proxy_sidstats/seed2.jsonl"),
]

for j in jobs:
    call = f.spawn(mode=j["mode"], seed=j["seed"], out_name=j["out_name"],
                    method="", stats_source=j["stats_source"], n_images=500)
    print(f"spawned {j['out_name']}: call_id={call.object_id}")

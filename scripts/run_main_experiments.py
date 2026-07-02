#!/usr/bin/env python
"""CARE full-table runner.

Runs `src/care.py` once per published cell (instance / style / celebrity, single- and
multi-concept), each with its own targets/nontargets/anchor-bank/gamma. Optionally
uploads each cell's report.json to a Hugging Face dataset repo as it completes (set
HF_TOKEN + C43A_HF_REPO); inert otherwise.
"""
import json, os, pathlib, shutil, subprocess, sys, threading, time, traceback

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
WORK = pathlib.Path(os.environ.get("C43A_HOME", "./.care_home"))
HF_REPO = os.environ.get("C43A_HF_REPO", "your-username/care-results")
HF_REPO_TYPE = "dataset"
HF_PREFIX = "c43a_results"
LOG_PATH = WORK / "run_log.txt"
PROGRESS_PATH = WORK / "progress.json"

A_INST  = ["Bugs Bunny","Hello Kitty","Garfield","Tom and Jerry","Donald Duck","Popeye"]
A_STYLE = ["Rembrandt","Vermeer","Michelangelo","Raphael","Botticelli","Titian"]
A_CELEB = ["Brad Pitt","Denzel Washington","Will Smith","Morgan Freeman","Keanu Reeves","Leonardo DiCaprio"]
NS = int(os.environ.get("C43A_NS", "10"))

def E(erase_type, targets, nontargets, anchors, gamma):
    return {"erase_type":erase_type,"targets":targets,"nontargets":nontargets,"anchors":anchors,
            "ns":NS,"rank":1,"op":"whitened","lam":0.5,"gamma":gamma}

# (iter_id, cfg) — each cell is run by invoking src/care.py once with ADAVD_CFG=cfg.
# The NSFW/I2P smoke-test cell is intentionally excluded here: it is not part of the
# paper's evaluation surface (see results/SCOREBOARD.md, "Excluded from this evaluation
# surface"), and its scorer lives outside this repo.
QUEUE = [
    ("iter001_inst_snoopy",
     E("instance", ["Snoopy"], ["Mickey Mouse","SpongeBob","Pikachu","Dog","Legislator"], A_INST, 0.5)),
    ("iter002_inst_snoopy_mickey",
     E("instance", ["Snoopy","Mickey Mouse"], ["SpongeBob","Pikachu","Dog","Legislator"], A_INST, 0.5)),
    ("iter003_inst_snoopy_mickey_spongebob",
     E("instance", ["Snoopy","Mickey Mouse","SpongeBob"], ["Pikachu","Dog","Legislator"], A_INST, 0.5)),
    ("iter004_style_vangogh",
     E("style", ["Van Gogh"], ["Picasso","Monet","Andy Warhol","Caravaggio"], A_STYLE, 0.2)),
    ("iter005_style_picasso",
     E("style", ["Picasso"], ["Van Gogh","Monet","Andy Warhol","Caravaggio"], A_STYLE, 0.2)),
    ("iter006_style_monet",
     E("style", ["Monet"], ["Van Gogh","Picasso","Andy Warhol","Caravaggio"], A_STYLE, 0.2)),
    ("iter007_celeb_brucelee",
     E("celebrity", ["Bruce Lee"], ["Marilyn Monroe","Melania Trump","Anne Hathaway","Tom Cruise"], A_CELEB, 0.2)),
    ("iter008_celeb_marilyn",
     E("celebrity", ["Marilyn Monroe"], ["Bruce Lee","Melania Trump","Anne Hathaway","Tom Cruise"], A_CELEB, 0.2)),
    ("iter009_celeb_melania",
     E("celebrity", ["Melania Trump"], ["Bruce Lee","Marilyn Monroe","Anne Hathaway","Tom Cruise"], A_CELEB, 0.5)),
]
SKIP_IF_DONE = True

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
    sys.stdout.write(line); sys.stdout.flush()
    with open(LOG_PATH, "a") as f: f.write(line)

def write_progress(state, iter_id="", extra=None):
    p = {"state": state, "iter_id": iter_id, "ts": time.time()}
    if extra: p.update(extra)
    PROGRESS_PATH.write_text(json.dumps(p, indent=2))

def _heartbeat():
    """Every C43A_HEARTBEAT_SEC (default 600s=10min) push a COMPACT live summary (the meaningful
    [c43a ...] log lines, not the MB of tqdm bars) + progress.json to HF, so a multi-hour cell's
    progress is visible without waiting for its report. Daemon thread; dies when main exits."""
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    interval = int(os.environ.get("C43A_HEARTBEAT_SEC", "600"))
    while True:
        time.sleep(interval)
        try:
            outs = sorted(WORK.glob("iter*.out"), key=lambda p: p.stat().st_mtime)
            if outs:
                lines = [l for l in open(outs[-1], errors="replace")
                         if any(k in l for k in ("[c43a", "[run]", "Traceback", "Error", "Killed"))]
                (WORK / "live_summary.txt").write_text(
                    f"live cell: {outs[-1].name}\nupdated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n" + "".join(lines[-80:]))
                api.upload_file(path_or_fileobj=str(WORK / "live_summary.txt"),
                                path_in_repo=f"{HF_PREFIX}/live_summary.txt", repo_id=HF_REPO, repo_type=HF_REPO_TYPE)
            if PROGRESS_PATH.exists():
                api.upload_file(path_or_fileobj=str(PROGRESS_PATH),
                                path_in_repo=f"{HF_PREFIX}/progress.json", repo_id=HF_REPO, repo_type=HF_REPO_TYPE)
        except Exception:
            pass

def upload_iter_artifacts(iter_id):
    """ONE upload per iter; retried (cheaply) on HF transients — never re-runs the kernel."""
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    out_dir = WORK / "outputs" / iter_id
    files = []
    for name in ("report.json", "results.json"):
        if (out_dir / name).exists(): files.append((out_dir / name, f"{HF_PREFIX}/{iter_id}/{name}"))
    if (WORK / f"{iter_id}.out").exists(): files.append((WORK / f"{iter_id}.out", f"{HF_PREFIX}/live/{iter_id}.out"))
    if LOG_PATH.exists(): files.append((LOG_PATH, f"{HF_PREFIX}/run_log.txt"))
    if PROGRESS_PATH.exists(): files.append((PROGRESS_PATH, f"{HF_PREFIX}/progress.json"))
    for src, dst in files:
        for attempt in range(4):
            try:
                api.upload_file(path_or_fileobj=str(src), path_in_repo=dst, repo_id=HF_REPO, repo_type=HF_REPO_TYPE)
                if dst.endswith("report.json") or dst.endswith("results.json"): log(f"  uploaded: {dst}")
                break
            except Exception as e:
                if attempt == 3: log(f"  upload err {dst}: {e!r}")
                else: time.sleep(30 * (attempt + 1))

def _done_valid(out_dir):
    """report.json only counts as 'done' if it's a real, full-template, non-skip result."""
    rp = out_dir / "report.json"
    if not rp.exists(): return False
    try: rep = json.loads(rp.read_text())
    except Exception: return False
    if str(rep.get("status", "")).startswith(("skipped", "failed", "error", "corrupt")): return False
    if int(rep.get("max_templates", 0)) > 0: return False   # a smoke (sliced) report — re-run for real
    return True

def run_iter(iter_id, cfg):
    out_dir = WORK / "outputs" / iter_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if SKIP_IF_DONE and _done_valid(out_dir):
        log(f"SKIP {iter_id} (valid report.json exists)")
        return None
    # Don't append onto a stale partial gen/ from a prior crash
    if (out_dir / "gen").exists():
        shutil.rmtree(out_dir / "gen", ignore_errors=True)

    env = os.environ.copy()
    env["ADAVD_CFG"] = json.dumps(cfg)
    env["ITER_ID"]   = iter_id
    env["C43A_HOME"] = str(WORK)
    env["C43A_REPO"] = str(WORK / "adavd_repo")
    env["C43A_SKIP_PIP"] = "1"
    # A stale smoke export must NOT silently truncate the real table
    if os.environ.get("C43A_SMOKE", "0") != "1":
        env["C43A_MAX_TEMPLATES"] = "0"
    cmd = [sys.executable, "-u", str(PROJECT_ROOT / "src" / "care.py")]

    log(f"=== START {iter_id} targets={cfg.get('targets')} gamma={cfg.get('gamma')} ===")
    write_progress("running", iter_id)
    if os.environ.get("HF_TOKEN", "").strip():
        try:   # push live 'running iterX' to HF at cell START so status is visible mid-run
            from huggingface_hub import HfApi
            HfApi(token=os.environ["HF_TOKEN"]).upload_file(path_or_fileobj=str(PROGRESS_PATH),
                path_in_repo=f"{HF_PREFIX}/progress.json", repo_id=HF_REPO, repo_type=HF_REPO_TYPE)
        except Exception:
            pass
    log_path = WORK / f"{iter_id}.out"
    timeout = int(os.environ.get("ITER_TIMEOUT", "0")) or None   # default: NO per-cell timeout (instance cells ~6h); set ITER_TIMEOUT only for an explicit cap

    # The kernel is non-resumable & multi-hour -> NEVER re-run it for an HF blip.
    # Run once; only retry if it died WITHOUT producing a report (i.e. not a completed run).
    rc = 1
    for attempt in range(int(os.environ.get("ITER_RETRIES", "1")) + 1):
        try:
            with open(log_path, "w") as logf:
                rc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = 124; log(f"  TIMEOUT {iter_id} after {timeout}s")
            break
        if rc == 0 or (out_dir / "report.json").exists():
            break
        try: tail = open(log_path).read()[-3000:]
        except Exception: tail = ""
        transient = any(x in tail for x in ("429 Client Error","Too Many Requests","504 Server Error",
                                            "Gateway Time-out","ConnectionError","ReadTimeout"))
        if not transient or attempt >= int(os.environ.get("ITER_RETRIES", "1")):
            break
        log(f"  transient startup err; retry in 120s"); time.sleep(120)

    log(f"=== END {iter_id} rc={rc} ===")
    write_progress("iter_done" if rc == 0 else "iter_failed", iter_id, {"rc": rc})
    if os.environ.get("HF_TOKEN", "").strip():
        try:
            upload_iter_artifacts(iter_id)
        except Exception as e:
            log(f"upload_iter_artifacts err: {e!r}")
    # Only drop images on a CLEAN cell (keep them for post-mortem on failure)
    if rc == 0 and (out_dir / "report.json").exists() and os.environ.get("C43A_KEEP_IMAGES", "0") != "1":
        shutil.rmtree(out_dir / "gen", ignore_errors=True); log(f"  freed gen/ for {iter_id}")
    return rc

def main():
    LOG_PATH.write_text("")
    # Contamination guard — anchors/targets/nontargets must be mutually disjoint
    for iid, c in QUEUE:
        t, n, a = set(c["targets"]), set(c["nontargets"]), set(c["anchors"])
        assert t.isdisjoint(n) and t.isdisjoint(a) and n.isdisjoint(a), f"{iid}: target/probe/anchor overlap"
    log(f"run_main_experiments.py starting — queue {len(QUEUE)}  NS={NS}  HF_REPO={HF_REPO}")
    log(f"config: C43A_SMOKE={os.environ.get('C43A_SMOKE','0')} C43A_MAX_TEMPLATES={os.environ.get('C43A_MAX_TEMPLATES','0')} "
        f"C43A_KEEP_IMAGES={os.environ.get('C43A_KEEP_IMAGES','0')} "
        f"ITER_TIMEOUT={os.environ.get('ITER_TIMEOUT','0') or 'unlimited'}")
    write_progress("starting")
    hf_enabled = bool(os.environ.get("HF_TOKEN", "").strip())
    if hf_enabled and int(os.environ.get("C43A_HEARTBEAT_SEC", "600")) > 0:
        threading.Thread(target=_heartbeat, daemon=True).start()
        log(f"heartbeat: every {os.environ.get('C43A_HEARTBEAT_SEC','600')}s -> c43a_results/live_summary.txt + progress.json")
    else:
        log("HF_TOKEN not set — running locally, no Hugging Face upload/heartbeat")

    only_iters = os.environ.get("ONLY_ITERS", "").strip()
    only_from  = os.environ.get("ONLY_FROM_ITER", "").strip()
    if only_iters:   # exact id OR exact iter-number token; mistype must hard-error
        wanted = set(s.strip() for s in only_iters.split(",") if s.strip())
        queue = [s for s in QUEUE if s[0] in wanted or s[0].split("_", 1)[0] in wanted]
        unmatched = [w for w in wanted if not any(s[0] == w or s[0].split("_", 1)[0] == w for s in QUEUE)]
        if unmatched: raise SystemExit(f"ONLY_ITERS matched nothing: {unmatched}")
        log(f"ONLY_ITERS={only_iters} -> {[s[0] for s in queue]}")
    elif only_from:
        queue, started = [], False
        for s in QUEUE:
            if s[0].split("_", 1)[0] == only_from or s[0] == only_from: started = True
            if started: queue.append(s)
        log(f"ONLY_FROM_ITER={only_from} -> {len(queue)} iters")
    else:
        queue = QUEUE
    if not queue: raise SystemExit("empty queue — nothing to run")

    iter_sleep = int(os.environ.get("INTER_ITER_SLEEP", "0"))
    fails, ran, skipped = [], 0, 0
    for spec in queue:
        try:
            rc = run_iter(*spec)
            if rc is None: skipped += 1; continue
            ran += 1
            if rc != 0:
                fails.append((spec[0], rc))
                if iter_sleep: log(f"sleep {iter_sleep*2}s after failure"); time.sleep(iter_sleep*2)
            elif iter_sleep:
                log(f"sleep {iter_sleep}s between iters"); time.sleep(iter_sleep)
        except Exception as e:
            log(f"EXCEPTION in {spec[0]}: {e!r}\n{traceback.format_exc()}")
            fails.append((spec[0], -1)); ran += 1
    log(f"=== CHAIN COMPLETE. ran={ran} skipped={skipped} fails={fails} ===")
    write_progress("chain_complete", extra={"ran": ran, "skipped": skipped, "fails": fails})

if __name__ == "__main__":
    main()

from pathlib import Path
import argparse,json,subprocess

ROOT=Path("/research/milsrg1/user_workspace/am3244/safety-alignment")
LITERATURE_DIR=ROOT/"scripts/initial_benchmark/almguard/literature"
OFFICIAL_REPO=LITERATURE_DIR/"ALMGuard"
OFFICIAL_URL="https://github.com/WeifeiJin/ALMGuard.git"
MANIFEST=ROOT/"data/almguard/attack_manifest.jsonl"

REQUIRED=[
    "README.md",
    "eval.py",
    "eval_qwen.py",
    "llm_evaluation.py",
    "utils.py",
]

def clone_official():
    LITERATURE_DIR.mkdir(parents=True,exist_ok=True)
    if OFFICIAL_REPO.exists():
        raise RuntimeError(f"Already exists: {OFFICIAL_REPO}")
    subprocess.run(
        ["git","clone","--depth","1",OFFICIAL_URL,str(OFFICIAL_REPO)],
        check=True,
    )

def validate_official():
    if not OFFICIAL_REPO.exists():
        raise FileNotFoundError(OFFICIAL_REPO)

    missing=[
        x for x in REQUIRED
        if not (OFFICIAL_REPO/x).exists()
    ]

    if missing:
        raise RuntimeError(
            "Incomplete official repo: "+", ".join(missing)
        )

    print("[OK] official ALMGuard source:",OFFICIAL_REPO)

def validate_manifest():
    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"{MANIFEST}\n"
            "The official ALMGuard repo does not ship all six "
            "attack-generation outputs. Build this manifest only after "
            "reproducing the attack inputs from the original attack sources."
        )

    n=0
    with MANIFEST.open("r",encoding="utf-8") as f:
        for ln,line in enumerate(f,1):
            if not line.strip():
                continue
            r=json.loads(line)
            n+=1
            p=ROOT/str(r["audio_path"])
            if not p.exists():
                raise FileNotFoundError(
                    f"{MANIFEST} line {ln}: {p}"
                )

    print("[OK] manifest rows:",n)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--clone-official",action="store_true")
    ap.add_argument("--validate-manifest",action="store_true")
    a=ap.parse_args()

    if a.clone_official:
        clone_official()

    validate_official()

    if a.validate_manifest:
        validate_manifest()

if __name__=="__main__":
    main()

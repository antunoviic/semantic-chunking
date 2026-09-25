#!/usr/bin/env bash
set -uo pipefail

SCRIPT="${BASH_SOURCE[0]}"
cd "$(dirname "$SCRIPT")/../.." || exit 1

export PYTHONUNBUFFERED=1
mkdir -p run_logs thesis/results

DL="${DOCS_DIR:-docs}"
CFG="--max-chunk-chars 1200 --max-chunk-sentences 100 --step-sentences 2"
PHASE="${PHASE:-0}"


DOCS=(
  "rfc9110|$DL/rfc9110.txt|eval_cache/rfc9110_questions_v3.json"
  "wells|$DL/wells.txt|eval_cache/wells_questions.json"
  "nasa|$DL/nasa.pdf|eval_cache/nasa_questions_literal_v3.json"
)

# arm name|cache suffix|flags
ARMS=(
  "no headings|incremental|--no-headings"
  "no filter|incremental_nofilter|--no-headings --no-filter"
  "line-based regex|incremental_headings_lines|--line-headings"
  "regex + LLM|incremental_headings_hybrid|--llm-headings"
)

SAMPLE_DOC_LABEL="${SAMPLE_DOC_LABEL:-nasa}"
SAMPLE_ARMS=(
  "midpoint split|incremental_midpoint|--no-headings --midpoint-split"
)

mem()  { vm_stat | awk '/Pages free/{gsub(/\./,"",$3); printf "%d MB free", $3*16384/1048576}'; }
disk() { df -h /System/Volumes/Data | tail -1 | awk '{print $4" disk free"}'; }

# chunking
chunk_doc() {
    local entry="$1" label doc stem
    IFS='|' read -r label doc _ <<< "$entry"
    [ -f "$doc" ] || { echo "!! missing: $doc"; return; }
    stem=$(basename "$doc"); stem="${stem%.*}"
    echo ""
    echo "---- Chunking: $label  ($(date '+%F %H:%M'))"

    arms=("${ARMS[@]}")
    if [ "$label" = "$SAMPLE_DOC_LABEL" ]; then
        arms+=("${SAMPLE_ARMS[@]}")
    fi

    for arm in "${arms[@]}"; do
        IFS='|' read -r armname suffix flags <<< "$arm"
        if [ "$suffix" = "window" ]; then
            cache="chunks_cache/${stem}.json"
        else
            cache="chunks_cache/${stem}_${suffix}.json"
        fi

        if [ -f "$cache" ] && [ -z "${FORCE_RECHUNK:-}" ]; then
            verdict=$(python - "$cache" <<'PY'
import json, sys
sys.path.insert(0, ".")
from app.chunk_cache import _code_digest
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("STALE|unreadable"); raise SystemExit
have, want = d.get("code_digest"), _code_digest()
if have == want:
    print(f"OK|{len(d['chunks'])} chunks, digest {have}")
else:
    print(f"STALE|digest {have or 'unstamped (old code)'} != {want}")
PY
)
            if [ "${verdict%%|*}" = "OK" ]; then
                echo "   skipped:   $label / $armname  (${verdict#*|})"
                continue
            fi
            echo "   stale:     $label / $armname  (${verdict#*|}) -> re-chunking"
            set -- --rechunk
        else
            set --
        fi

        log="run_logs/v4_chunk_${label}_${suffix}.log"
        echo ""
        echo "-> $label / $armname   $(date '+%H:%M')   $(mem), $(disk)"
        # shellcheck disable=SC2086
        python -m app.main "$doc" $CFG $flags --chunk-only "$@" ${FORCE_RECHUNK:+--rechunk} > "$log" 2>&1
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "   !! aborted (exit $rc) — see $log"
            tail -3 "$log"
            continue
        fi
        grep -E "boundary sources|Saved" "$log" | sed 's/^/   /'
    done
}

# derived arms

enrich_doc() {
    local entry="$1" label doc
    IFS='|' read -r label doc _ <<< "$entry"
    [ -f "$doc" ] || return
    echo ""
    echo "---- Enrichment: $label  ($(date '+%F %H:%M'))"
    python thesis/scripts/enrich_arm.py "$doc" 2>&1 \
        | grep -vE "^\\[cache\\] (Loaded|Old)" | sed 's/^/   /'
}

# evaluation
eval_doc() {
    local entry="$1" label doc questions log rc
    IFS='|' read -r label doc questions <<< "$entry"
    [ -f "$doc" ] || return
    [ -f "$questions" ] || { echo "!! question set missing: $questions"; return; }
    echo ""
    echo "---- Evaluation: $label  ($(date '+%F %H:%M'))"

    log="run_logs/v4_eval_${label}.log"
    echo ""
    echo "-> $label   $(date '+%H:%M')   $(mem)"

    python -m app.main "$doc" $CFG --no-headings --questions-file "$questions" > "$log" 2>&1
    rc=$?
    [ $rc -ne 0 ] && { echo "   !! aborted (exit $rc) — see $log"; tail -3 "$log"; return; }
    grep -E "^\| (llm|fixed|recursive|semantic)" "$log" | sed 's/^/   /'
}

echo "=========================================================="
echo "RUN v4  ($(date '+%F %H:%M'))   phase: ${PHASE} (0 = chunk + evaluate)"
echo "=========================================================="
for entry in "${DOCS[@]}"; do
    if [ "$PHASE" = "0" ] || [ "$PHASE" = "1" ]; then
        chunk_doc "$entry"
        [ -n "${SKIP_ENRICH:-}" ] || enrich_doc "$entry"
    fi
    if [ "$PHASE" = "0" ] || [ "$PHASE" = "2" ]; then eval_doc "$entry"; fi
done

# ------------------------------------------------- provenance check
echo ""
echo "=========================================================="
echo "PROVENANCE — do all arms come from the same code state?"
echo "=========================================================="
# Only the arms THIS run produces. chunks_cache/ also holds arms of documents
# that were dropped and of arms that were retired; scanning those made the
# check report a split code state on every run — noise exactly where it must
# not be. The expected names are derived from DOCS and ARMS, so the check
# follows the run instead of drifting from it.
EXPECT=""
for entry in "${DOCS[@]}"; do
    IFS='|' read -r label doc _ <<< "$entry"
    stem=$(basename "$doc"); stem="${stem%.*}"
    arms=("${ARMS[@]}")
    [ "$label" = "$SAMPLE_DOC_LABEL" ] && arms+=("${SAMPLE_ARMS[@]}")
    [ -n "${SKIP_ENRICH:-}" ] || arms+=("enrichment|incremental_enriched|")
    for arm in "${arms[@]}"; do
        IFS='|' read -r _ suffix _ <<< "$arm"
        if [ "$suffix" = "window" ]; then
            EXPECT="$EXPECT ${stem}.json"
        else
            EXPECT="$EXPECT ${stem}_${suffix}.json"
        fi
    done
done
export RUN_EXPECT="$EXPECT"
python - <<'PYEOF'
import json, os, pathlib, collections
expect = [n for n in os.environ.get("RUN_EXPECT", "").split() if n]
revs = collections.defaultdict(list)
pythons = set()
missing = []
for name in expect:
    p = pathlib.Path("chunks_cache") / name
    if not p.exists():
        missing.append(name)
        continue
    try:
        d = json.load(open(p))
    except Exception:
        missing.append(f"{name} (unreadable)")
        continue
    # Group by digest ONLY. The interpreter is shown per file, not folded into
    # the key: caches written before the python stamp existed have no tag, and
    # grouping on it would report them as a second code state.
    rev = d.get("code_digest", "unstamped")
    py = d.get("python")
    pythons.add(py)
    revs[rev].append(f"{name}  ({len(d.get('chunks', []))} chunks)"
                     + (f"   [Python {py}]" if py else ""))
for rev, files in sorted(revs.items()):
    print(f"\n  {rev}   — {len(files)} of {len(expect)} arms")
    for f in files:
        print(f"      {f}")
if missing:
    print(f"\n  not yet chunked — {len(missing)}")
    for m in missing:
        print(f"      {m}")
print()
known = {p for p in pythons if p}
if len(revs) > 1:
    print("  !! More than one code state among the arms of this run. Arms from")
    print("     different states are not comparable — re-chunk the affected ones.")
elif missing:
    print("  Arms present are from one code state; the ones listed above are absent.")
else:
    print("  OK — every arm of this run comes from one code state.")
if len(known) > 1:
    print(f"  !! Arms were chunked under different Python versions: "
          f"{', '.join(sorted(known))}. The digest is derived from the AST, whose")
    print("     text form is version-dependent, so this needs checking by hand.")
elif known and None in pythons:
    print(f"  (Some arms predate the interpreter stamp; the stamped ones say "
          f"Python {known.pop()}.)")
PYEOF

echo ""
echo "Done: $(date '+%F %H:%M')"

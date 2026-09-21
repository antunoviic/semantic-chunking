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
python - <<'PY'
import json, glob, pathlib, collections
revs = collections.defaultdict(list)
for p in sorted(glob.glob("chunks_cache/*.json")):
    name = pathlib.Path(p).name
    if name.count(".") > 1:          # skip timestamped backups
        continue
    try:
        d = json.load(open(p))
    except Exception:
        continue
    rev = d.get("code_digest", "unstamped")
    revs[rev].append(f"{name}  ({len(d.get('chunks', []))} chunks)")
for rev, files in revs.items():
    print(f"\n  {rev}   — {len(files)} caches")
    for f in files:
        print(f"      {f}")
if len(revs) > 1:
    print("\n  !! More than one code state. Arms from different states are not")
    print("     comparable — re-chunk the affected ones.")
PY

echo ""
echo "Done: $(date '+%F %H:%M')"

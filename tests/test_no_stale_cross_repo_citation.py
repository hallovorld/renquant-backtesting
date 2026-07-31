"""A correction that lands in one repo must not leave its quote standing in another.

`orch#680` published *"ten of eleven cited artifacts could not be re-derived from the
files they name"*, and this repo's progress doc cited it as supporting evidence. That
finding was **retracted** in `orch@bf8eb40e` — it came from reading the legacy top-level
`wf_gate_metadata` instead of the canonical `metadata.wf_gate_metadata`, the same key
confusion this document already retracts one section above. Re-measured, all eleven
artifacts carry the canonical block and all eleven regime blocks are re-derivable.

The retraction landed in `renquant-orchestrator`. **The citation lived here**, and did
not move. That is the sixth instance on this programme of a claim outliving its own
correction, and the first that crosses a repository boundary — which is precisely why
neither repo's own review caught it.

This test is deliberately narrow: it pins the retracted sentences so they cannot come
back unmarked. It cannot detect the general case (this repo has no view of another
repo's retractions), and pretending otherwise would be the over-claim this whole
document is about.
"""

from __future__ import annotations

import pathlib
import re

DOC = (pathlib.Path(__file__).resolve().parent.parent
       / "doc" / "progress" / "2026-07-31-wf-gate-booster-blind.md")

#: Sentences withdrawn on 2026-07-31. Each may appear ONLY inside a strikethrough span.
WITHDRAWN = (
    "ten of eleven cited artifacts could not be re-derived",
    "nothing on this path scores the candidate's own weights",
)


def _flat(text: str) -> str:
    """Blockquotes and hard wraps split these sentences across lines, so match on the
    flattened text. A line-oriented check missed exactly this on orch#676."""
    return re.sub(r"\s*\n>?\s*", " ", text)


def test_every_withdrawn_sentence_sits_inside_a_strikethrough():
    text = _flat(DOC.read_text(encoding="utf-8"))
    struck = [m.span() for m in re.finditer(r"~~.+?~~", text, re.S)]
    assert struck, "no strikethrough span at all — no withdrawal is marked"
    for phrase in WITHDRAWN:
        hits = [m.start() for m in re.finditer(re.escape(phrase), text)]
        assert hits, f"the withdrawal itself must stay on the record: {phrase!r}"
        for at in hits:
            assert any(a <= at < b for a, b in struck), \
                f"withdrawn claim standing unmarked at offset {at}: {phrase!r}"


def test_the_retraction_names_the_cause_and_the_corrected_measurement():
    """A withdrawal that does not say WHY reads as a change of mind. This one has a
    mechanical cause (wrong key) and a replacement number, and both must be stated or
    the next reader repeats it — as I did, three times in one day."""
    text = DOC.read_text(encoding="utf-8")
    assert "metadata.wf_gate_metadata" in text
    assert "legacy top-level" in text
    assert re.search(r"all eleven .{0,40}re-derivable", text, re.I | re.S)


def test_the_surviving_point_does_not_depend_on_the_withdrawn_example():
    """Anti-vacuity for the two tests above: striking a sentence is only half a
    correction. The paragraph must still state the claim it was there to support —
    otherwise the withdrawal quietly deletes a conclusion that was never wrong."""
    text = _flat(DOC.read_text(encoding="utf-8"))
    struck_spans = [m.span() for m in re.finditer(r"~~.+?~~", text, re.S)]
    tail = text[text.index("immutable source snapshot"):]
    surviving = "an artifact store is not"
    at = tail.find(surviving)
    assert at >= 0, "the digest argument was deleted along with its example"
    abs_at = text.index(tail) + at
    assert not any(a <= abs_at < b for a, b in struck_spans), \
        "the surviving argument must not itself be struck through"

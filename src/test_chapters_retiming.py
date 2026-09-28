'''Chapters ride the repaired temporary file (Addendum 29 closed by the owner, 2026-09-28) --
run: python3 src/test_chapters_retiming.py [--real-only] (pytest also runs it).

`merge_video_chimeric.mux_repaired_file` never copies chapters from its inputs
(`-map_chapters -1`, `build_one_audio_track`'s comment) -- the ONLY chapters the repaired
temporary carries are `mux_chapters`'s single call on `build_delivered_chapters`'s retimed XML
(`repair_orchestrator.py` step 5, once, before the build step; `merge_video_repair.py:411` /
`merge_video_chimeric.py:2834` thread `chapters_path` through unchanged). This file is the
regression test that proposal draft (INTEGRITY_HOOKS_PROPOSAL_20260925.md s6) and the LAB
(architect/cases/LAB_chapters_route_A_20260928.md) asked for and neither shipped: a synthetic
candidate with a head trim + an interior cut, and one real pair (errid-675, corpus).
Every source file is READ-ONLY; everything written goes to a temporary directory, removed.'''

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ElementTree
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tools  # noqa: E402

if not getattr(tools, "software", None):
    tools.software = {}
for _tool in ("ffmpeg", "ffprobe", "mkvmerge", "mkvextract"):
    tools.software.setdefault(_tool, _tool)
if not getattr(tools, "tmpFolder", None) or tools.tmpFolder == "/tmp":
    tools.tmpFolder = tempfile.mkdtemp(prefix="test_chapters_")

import merge_video_chimeric as mvc  # noqa: E402

TOOLS = all(shutil.which(tools.software[t]) for t in ("ffmpeg", "ffprobe", "mkvmerge",
                                                       "mkvextract"))
E675 = "/home/vmsam/src/VMSAM_CORPUS/video-pairs/errid-675"


def _chapters_xml(entries):
    '''entries: [(start_ms, title)] -> a minimal, valid Matroska chapters XML body (no edition
    UID: mkvmerge assigns its own, as `build_delivered_chapters` already relies on for a
    candidate edition).'''
    atoms = "".join(
        f"    <ChapterAtom>\n      <ChapterTimeStart>{mvc.format_chapter_time(Decimal(ms))}"
        f"</ChapterTimeStart>\n      <ChapterDisplay><ChapterString>{title}</ChapterString>"
        f"<ChapterLanguage>eng</ChapterLanguage></ChapterDisplay>\n    </ChapterAtom>\n"
        for ms, title in entries)
    return ("<?xml version=\"1.0\"?>\n<Chapters>\n  <EditionEntry>\n"
            "    <EditionFlagOrdered>0</EditionFlagOrdered>\n" + atoms + "  </EditionEntry>\n"
            "</Chapters>\n")


def _mux_with_chapters(tmp, name, chapters_entries, duration_s=1):
    '''A tiny real MKV (silent audio, `duration_s`) carrying `chapters_entries` -- built the
    same way as any real ripped file: base media, then `mkvmerge --chapters`.'''
    base = os.path.join(tmp, f"{name}_base.mkv")
    subprocess.run([tools.software["ffmpeg"], "-nostdin", "-v", "error", "-y", "-f", "lavfi",
                    "-i", f"anullsrc=r=8000:cl=mono", "-t", str(duration_s), "-c:a", "flac",
                    base], check=True)
    if not chapters_entries:
        return base
    xml_path = os.path.join(tmp, f"{name}_chapters.xml")
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(_chapters_xml(chapters_entries))
    out = os.path.join(tmp, f"{name}.mkv")
    subprocess.run([tools.software["mkvmerge"], "-q", "-o", out, "--chapters", xml_path, base],
                   check=True)
    return out


def _atoms_of_root(root):
    out = []
    for e, edition in enumerate(root.findall("EditionEntry")):
        for atom in edition.findall("ChapterAtom"):
            start = mvc.parse_chapter_time_ms(atom.findtext("ChapterTimeStart"))
            title = atom.findtext("ChapterDisplay/ChapterString")
            out.append((e, int(start), title))
    return out


def _chapter_atoms(mkv_path):
    '''[(edition_index, start_ms, title)] of `mkv_path`, via mkvextract + the module's own
    `parse_chapter_time_ms` (the production parser, not a second implementation).'''
    xml_path = mkv_path + ".extracted.xml"
    root, reason = mvc.extract_chapters_xml(mkv_path, xml_path)
    if root is None:
        return [], reason
    return _atoms_of_root(root), "extracted"


def _chapter_atoms_of_xml(xml_path):
    '''Same as `_chapter_atoms`, but `xml_path` is already a chapters XML (`build_delivered_
    chapters`'s own output, not a muxed MKV -- mkvextract only reads containers).'''
    return _atoms_of_root(ElementTree.parse(xml_path).getroot())


class Synthetic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not TOOLS:
            raise unittest.SkipTest("ffmpeg/mkvmerge/mkvextract missing")
        cls.tmp = tempfile.mkdtemp(prefix="test_chapters_syn_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_head_trim_and_interior_cut_retime_the_chapters_once(self):
        # candidate: 3 chapters at 0 s, 60 s, 150 s (its OWN timeline). Plan: a 10 s head trim
        # (candidate [0;10) cut) then content to 80 s, an interior cut removing candidate
        # [80;100), content resumes to 200 s -- exactly the head-offset + one cut/fill shift the
        # owner asked for, same piecewise mapping subtitle cues already use
        # (`map_candidate_time_to_master`).
        candidate = _mux_with_chapters(self.tmp, "cand",
                                       [(0, "C1"), (60_000, "C2"), (150_000, "C3")])
        master = _mux_with_chapters(self.tmp, "mstr", [])       # no master chapters: simplest
        reference_pieces = [
            {"source": "candidate", "source_start_ms": "10000",
             "master_start_ms": "0", "master_end_ms": "70000"},        # candidate [10;80) -> [0;70)
            {"source": "candidate", "source_start_ms": "100000",
             "master_start_ms": "70000", "master_end_ms": "170000"},   # candidate [100;200) -> [70;170)
        ]
        chapters_path, decisions = mvc.build_delivered_chapters(
            master, candidate, reference_pieces, 1, 170_000, self.tmp)
        self.assertIsNotNone(chapters_path, decisions)
        atoms = _chapter_atoms_of_xml(chapters_path)
        # count unchanged: 3 in, 3 out, one edition (no master chapters to keep alongside)
        self.assertEqual(len(atoms), 3, atoms)
        self.assertEqual({e for e, _, _ in atoms}, {0})
        got = [(start, title) for _, start, title in atoms]
        expected = [(0, "C1"),            # was cut away (0 < 10 s): snapped to the next piece
                    (50_000, "C2"),        # 60 s - 10 s head offset
                    (120_000, "C3")]       # 150 s - 100 s + 70 s (the second piece's own start)
        self.assertEqual(got, expected)
        decided = {(d.get("atom"), d.get("start")) for d in decisions if d.get("source") == "candidate"
                  and "atom" in d}
        self.assertIn(("C1", "snapped_to_next_piece"), decided)
        self.assertIn(("C2", "mapped"), decided)
        self.assertIn(("C3", "mapped"), decided)

        # NOW the temporary file itself: `mux_repaired_file` never copies the inputs' chapters
        # (`-map_chapters -1`) -- the retimed XML is the only source, applied once.
        report = [{"path": candidate, "language": "eng", "title": None, "marker": "chimeric"}]
        out_path = os.path.join(self.tmp, "repaired.mkv")
        mvc.mux_repaired_file(report, [], out_path, "chimeric", 60, "test",
                              chapters_path=chapters_path)
        self.assertTrue(os.path.exists(out_path))
        out_atoms, out_reason = _chapter_atoms(out_path)
        self.assertEqual(out_reason, "extracted")
        self.assertEqual(sorted(out_atoms), sorted([(e, s, t) for e, s, t in
                                                    [(0, 0, "C1"), (0, 50_000, "C2"),
                                                     (0, 120_000, "C3")]]))
        # `mux_chapters` is NOT idempotent -- mkvmerge ADDS an edition, it does not replace one
        # already on the file (measured here) -- which is exactly why `mux_repaired_file` must
        # call it AT MOST ONCE (grep of merge_video_chimeric.py: the only call site is line
        # ~2834, `assemble_or_log_the_decline`'s single mux). A second call on an already-
        # chaptered file doubles the editions instead of leaving them alone.
        mvc.mux_chapters(out_path, chapters_path, 60)
        again, _ = _chapter_atoms(out_path)
        self.assertEqual(len(again), 6, again)
        self.assertEqual({e for e, _, _ in again}, {0, 1})

    def test_candidate_without_chapters_delivers_none(self):
        candidate = _mux_with_chapters(self.tmp, "cand_nochap", [])
        master = _mux_with_chapters(self.tmp, "mstr_nochap", [])
        chapters_path, decisions = mvc.build_delivered_chapters(
            master, candidate, [], 1, 10_000, self.tmp)
        self.assertIsNone(chapters_path)
        reasons = {d["source"]: d["extraction"] for d in decisions if "extraction" in d}
        self.assertEqual(reasons, {"master": "no_chapters", "candidate": "no_chapters"})


@unittest.skipUnless(TOOLS and os.path.exists(os.path.join(E675, "candidate.mkv")),
                     "errid-675 corpus pair or tools absent")
class RealPair(unittest.TestCase):
    '''errid-675: candidate carries exactly 3 chapters (Prologue 0 s, Intro 157 s, Part 1
    243 s), the master carries 2 editions (6 + 2 atoms) -- MEASURED via mkvextract, 2026-09-28.
    A 7 s head trim (illustrative: the real repair's own plan is not re-run here, ADDENDUM 29's
    ask is the mechanism, not this pair's true offset) puts them at 0 s (snapped), 150 s, 236 s.'''

    def test_real_candidate_chapters_survive_retimed_once(self):
        master = os.path.join(E675, "master.mkv")
        candidate = os.path.join(E675, "candidate.mkv")
        tmp = tempfile.mkdtemp(prefix="test_chapters_e675_")
        try:
            reference_pieces = [{"source": "candidate", "source_start_ms": "7000",
                                 "master_start_ms": "0", "master_end_ms": "1440000"}]
            chapters_path, decisions = mvc.build_delivered_chapters(
                master, candidate, reference_pieces, 1, 1_440_000, tmp)
            self.assertIsNotNone(chapters_path)
            atoms = _chapter_atoms_of_xml(chapters_path)
            # master's 2 editions (6 + 2 atoms) kept as-is, plus the candidate's 1 retimed
            # edition (3 atoms): 3 editions, 11 atoms total -- none duplicated
            self.assertEqual(len(atoms), 11, atoms)
            self.assertEqual({e for e, _, _ in atoms}, {0, 1, 2})
            candidate_atoms = sorted((s, t) for e, s, t in atoms if e == 2)
            self.assertEqual(candidate_atoms,
                             [(0, "Prologue"), (150_000, "Intro"), (236_000, "Part 1")])

            # the real 5 s slice of the candidate's own audio, muxed the way `mux_repaired_file`
            # really does it (`-map_chapters -1` -- no input chapters ride along), chapters
            # applied once by `mux_chapters`: the temporary lists exactly this XML.
            report = [{"path": candidate, "language": "jpn", "title": None, "marker": "chimeric"}]
            out_path = os.path.join(tmp, "repaired.mkv")
            mvc.mux_repaired_file(report, [], out_path, "chimeric", 120, "test",
                                  chapters_path=chapters_path)
            out_atoms, out_reason = _chapter_atoms(out_path)
            self.assertEqual(out_reason, "extracted")
            self.assertEqual(len(out_atoms), 11, out_atoms)
            out_candidate_atoms = sorted((s, t) for e, s, t in out_atoms if e == 2)
            self.assertEqual(out_candidate_atoms, candidate_atoms)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def main(argv):
    loader = unittest.TestLoader()
    classes = [Synthetic] if "--real-only" not in argv else [RealPair]
    if "--real-only" not in argv:
        classes.append(RealPair)
    suite = unittest.TestSuite()
    for c in classes:
        suite.addTests(loader.loadTestsFromTestCase(c))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

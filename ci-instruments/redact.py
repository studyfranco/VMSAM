#!/usr/bin/env python3
"""Redact a log so it can leave this machine. USE THIS, never an ad-hoc regex.

Written after redacting the same class of line twice by hand and leaking a title
BOTH TIMES -- the first regex stopped at a path separator, the second stopped at a
quote. The failure is not the pattern, it is hand-writing a pattern per excerpt.

WHITELIST, NOT BLACKLIST. A blacklist removes what you thought of; a whitelist keeps
what you can name. The rule: a reader with the library must not resolve an excerpt to
a title. A filename resolves in zero lookups, a catalogue id in one, a frame in none.
"""
import re, sys

KEEP_CODE = re.compile(r'^/home/vmsam/[\w/]+\.py$')       # code paths are not media

def redact(text):
    out = []
    for line in text.splitlines():
        # 1. any absolute media path, greedy to end of the path-ish run
        line = re.sub(r'/srv/[^\s\'"\]},]*', '<media path>', line)
        line = re.sub(r'/config/output/[^\s\'"\]},]*', '<test output>', line)
        # 2. catalogue ids in any bracket form
        line = re.sub(r'[\[\{]\s*tvdb[^\]\}]*[\]\}]', '', line, flags=re.I)
        line = re.sub(r'\b(tvdb|tmdb|imdb|anilist)[-_ ]?id?[-_ ]?\d+', '', line, flags=re.I)
        # 3. any filename with a media extension
        line = re.sub(r'[^\s\'"/\]]+\.(mkv|mp4|avi|m2ts|flac|ass|srt)\b', '<file>', line, flags=re.I)
        # 4. residual title runs: a bare dotted or spaced name before Sxx Exx
        line = re.sub(r'[\w.\'\- ]+?[ .]S\d{2}E\d{2}[\w.\- ]*', '<episode>', line)
        # 5. anything still carrying a capitalised multiword run next to a season marker
        line = re.sub(r'[\w.\'\- ]{2,}/Season \d+[^\s\'"\]}]*', '<season>', line)
        out.append(line)
    return "\n".join(out)


# THE SELF-CHECK MUST RUN ON EVERY USE, NOT ONLY FROM THE COMMAND LINE.
# It lived under `if __name__ == "__main__"`, so importing redact() as a function
# walked straight past it -- and a bare filename carrying a release group and an
# episode number ("[Group] Title - 24 [1080p][HASH].mkv") survived into a file on
# disk. It has no /Season N/ and no SxxExx, so the path rules never saw it; the
# self-check WOULD have caught it on the media extension and refused to emit.
# That is the import-side-effect defect inverted: the guard was behind main, and
# I imported past the guard. Fifth instance of that shape tonight.
BARE = [
    (r'\[[^\]\/]{2,30}\][^\/\n]{0,80}?\.(mkv|mp4|avi|m4v|mka|ass|srt)', '<file>'),
    (r'[A-Za-z0-9][\w\'". -]{3,80}?\s-\s\d{1,4}\s*\[[^\]]*\][^\/\n]*?\.(mkv|mp4|avi|m4v|mka|ass|srt)', '<file>'),
    (r'[\w\'". ()-]{4,90}?\.(mkv|mp4|avi|m4v|mka|ass|srt)', '<file>'),
]


# SECOND LEAK, SAME SHAPE AS THE FIRST. A title survived as
#   "<media path> Strange Fake   /Specials/<file>"
# because the season rule matched "/Season N/" and this tree uses "/Specials/".
# The general defect is that the rules enumerate the FORMS a path takes, and a
# library invents new ones. So the last rule is now structural rather than
# enumerative: ANYTHING left standing between a redacted path token and the next
# redacted token is itself path, whatever it looks like.
STRUCTURAL = [
    (r'(<(?:media path|season|file|episode|PATH)>)[^\n<]{1,200}?(?=<(?:media path|season|file|episode|PATH)>)',
     r'\1<path>'),
    (r'/(?:Specials|Season[^/\n]*|Movies|Anime|Series|TV[^/\n]*)/[^\s\'"\]}]*', '<season>'),
    (r'(<media path>)[^\n<]{1,200}$', r'\1<path>'),
    # the check caught a capitalised multiword run before a separator and there was
    # NO RULE THAT REMOVED IT -- a guard that can detect what it cannot fix just
    # refuses forever. Detection and repair have to be added together.
    (r"(?:[A-Z][\w']{1,20}[ ]){1,6}[A-Z][\w']{1,20}\s*/", '<title>/'),
]

def redact_checked(text):
    """redact(), then REFUSE to return if anything resolvable survives."""
    out = redact(text)
    for pat, rep in STRUCTURAL:
        out = re.sub(pat, rep, out)
    for pat, rep in BARE:
        out = re.sub(pat, rep, out)
    bad = []
    for pat, why in [(r'/srv/\w', 'library path'), (r'tvdb[a-z]*-?\d', 'catalogue id'),
                     (r'S\d{2}E\d{2}', 'episode marker'),
                     (r'[\w)\]]\s*\.(mkv|mp4|avi|m4v|mka|ass|srt)\b', 'media filename'),
                     # a capitalised multiword run next to a path separator is a title
                     (r'(?:[A-Z][\w\']{1,20}[ ]){1,6}[A-Z][\w\']{1,20}\s*/', 'title beside a separator'),
                     # SAME RULE FOR THE OTHER SEPARATOR. Every rule above keys on '/',
                     # so a backslash path walked straight through the CHECK -- not just
                     # the redactor. Controlled: 'D:\\Media\\Some Title \\Season 1\\f.mkv'
                     # was EMITTED with Media, Some Title and Season intact. The corpus is
                     # POSIX so this was not a live leak, but the checker's whole job is to
                     # refuse rather than to be right about the shape it will meet, and it
                     # cannot do that for a separator it does not know about.
                     (r'(?:[A-Z][\w\']{1,20}[ ]){1,6}[A-Z][\w\']{1,20}\s*\\\\?', 'title beside a backslash'),
                     (r'[A-Za-z]:\\\\', 'drive-letter path'),
                     (r'\\\\(?:Media|Season|Specials|Movies|Anime|Series)\\\\?', 'library folder, backslash'),
                     (r'/(?:Specials|Movies|Anime|Series)/', 'library folder')]:
        # CASE SENSITIVITY IS PART OF THE PATTERN, NOT A GLOBAL. Applying re.I to
        # every check turned the title rule -- which keys on CAPITALISATION -- into
        # one that fires on ordinary lowercase prose before a slash. It refused a log
        # containing "invalid data found/" and nothing resembling a title. An alarm
        # that fires on sound cases is the failure mode that gets a guard switched
        # off, and I built one into the guard within an hour of naming it.
        flags = 0 if why in ('title beside a separator',) else re.I
        if re.search(pat, out, flags):
            bad.append(why)
    if bad:
        raise ValueError("REFUSING TO EMIT -- survived redaction: " + ", ".join(sorted(set(bad))))
    return out

if __name__ == "__main__":
    src = open(sys.argv[1], encoding='utf-8', errors='replace').read()
    txt = redact(src)
    # SELF-CHECK: refuse to emit if anything suspicious survives.
    # REQUIRE AN ACTUAL PATH, not the bare prefix. The first version flagged a line
    # DESCRIBING this very pattern -- a guard that cries wolf on correct text gets
    # overridden, which is worse than no guard. Same lesson as the instrument-signature
    # guard that keyed on a field varying by design.
    bad = [p for p in (r'/srv/\w', r'tvdb[-_ ]?\d', r'[\w.\-]+\.mkv\b',
                       r'\bS\d{2}E\d{2}\b')
           if re.search(p, txt, re.I)]
    if bad:
        sys.exit(f"REFUSING TO EMIT -- survivors: {bad}")
    print(txt)


def safe_text(s, limit=200):
    """THE CENSOR AT THE CONSTRUCTION SITE, not at the write.

    Two leaks reached artefacts today and BOTH WERE FREE-TEXT REASON FIELDS -- a `why`
    and a `decline_msg`. Not one structured field leaked: those were scrubbed because
    somebody thought about them, and nobody constructs a reason string thinking about
    privacy. They construct it thinking about being understood.

    A REASON FIELD INHERITS THE VOCABULARY OF WHATEVER PRODUCED IT. A decline message
    carrying a filename means it was built by interpolating an exception, a log line or
    a path variable that already contained one. THE FIELD IS THE EXIT, NOT THE SOURCE --
    so auditing fields forever is the wrong list, and censoring where reasons are
    ASSEMBLED is a small and permanent one.

    Refuses rather than emits: an unrenderable value becomes a stated placeholder, so
    the FACT survives and the value does not.
    """
    try:
        return redact_checked(str(s)[:limit])
    except ValueError:
        return ("<REDACTED: carried a library path and could not be safely rendered; "
                "the fact is kept, the value is not>")

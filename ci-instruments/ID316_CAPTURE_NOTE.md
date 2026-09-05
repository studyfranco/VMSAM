# id 316 — what to record AT THE MOMENT OF CAPTURE, and why

**316 is in this run's `ONLY_IDS` and has not been reached.** When it lands, its
locator/pipeline comparison must be recorded **NOT AN AGREEMENT TEST**, marked at the
point of capture rather than retrospectively.

## The reason, which travels with the marking

`vmsam-ci-pair`, measured with `ffprobe -select_streams a -show_entries stream=sample_rate`
**on the files themselves** — `/errors` carries no rate field at all:

    id 316   folder 177   audio streams: 32000, 44100
    id 307   folder 176   audio streams: 32000, 44100
    population: their column 1, 315 entries, 315 readable, container 5690f31

**2 files of 315 carry a sub-44100 stream. They are 307 and 316. There is no third.**

dev-1's locator pins `-ar 44100`; the pipeline uses `get_less_sampling_rate(pair)` and
clamps only when the lower rate is ABOVE 44100. On these two files the lower rate is
**32000**, so the pipeline goes low and the locator upsamples. **The two instruments are
12.1 kHz apart — not a rounding, not a tolerance question, two different signals.**
Neither instrument shows this in its numbers.

## Why the reason is written down and not just the instruction

"Suspended per the Lead" expires the moment the reasoning is forgotten. **"The two
instruments resampled to rates 12.1 kHz apart" is checkable forever.**

    A MARKING THAT CARRIES ITS REASON SURVIVES THE PERSON WHO MADE IT.

Agreement on these two files would be luck and disagreement would be unattributable
between grid and defect. **Record the numbers; do not record a verdict.**

**307 is outside this run's population entirely and I will not be a second opinion on it.**

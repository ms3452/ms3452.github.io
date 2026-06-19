# Methodology

How the arXiv `physics.optics` vs. Nature Photonics title analysis is built — from
the raw 5 GB dump to the figures in [`explore_titles.ipynb`](explore_titles.ipynb).
A companion notebook, [`browse_titles.ipynb`](browse_titles.ipynb), is a manual
explorer for the same data (search titles, filter by year, follow links to the
articles); it is described in [§4](#4-browsing-the-raw-titles-browse_titlesipynb).

---

## 1. Datasets & extraction pipeline

All heavy extraction lives in [`raw-data/`](raw-data/) and writes small CSVs that the
notebook reads.

| Output | Produced by | Columns | Rows |
|---|---|---|---|
| [`physics_optics_titles.csv`](physics_optics_titles.csv) | [`raw-data/filter_optics.py`](raw-data/filter_optics.py) | `year, title, categories, id, doi, url` | 56,581 |
| [`raw-data/arxiv_all.csv`](raw-data/arxiv_all.csv) | [`raw-data/extract_all_arxiv.py`](raw-data/extract_all_arxiv.py) | `year, categories` | 3,073,376 |
| [`nature_photonics_titles.csv`](nature_photonics_titles.csv) | [`raw-data/make_nphoton_master.py`](raw-data/make_nphoton_master.py) | `year, title, doi, url` | 5,149 |
| `raw-data/nphoton_<year>.csv` | [`raw-data/scrape_nphoton_2026.py`](raw-data/scrape_nphoton_2026.py) | `published, type, title, authors, doi, url` | per year |

**Source data**

- **arXiv** — `raw-data/arxiv-metadata-oai-snapshot.json` (the ~5 GB Kaggle dump, JSON
  Lines: one JSON object per line). Streamed line-by-line so memory stays tiny.
  - `filter_optics.py` keeps records whose space-separated `categories` field
    **contains** `physics.optics` (i.e. primary *or* cross-listed) and emits the title,
    full category list, arXiv `id`, publisher `doi`, and a `url`.
  - `extract_all_arxiv.py` keeps **every** record (year + categories only) for the
    growth comparison.
- **Nature Photonics** — pulled from the **Crossref REST API** by ISSN `1749-4893`
  (`scrape_nphoton_2026.py`), one CSV per year 2007–2026, then concatenated into a single
  master by `make_nphoton_master.py`.

**Article links.** Both title masters carry a clickable `url` so any row can be
opened directly. For arXiv it is the abstract page `https://arxiv.org/abs/<id>`,
which always exists (the `doi` column additionally holds the publisher DOI when
arXiv records one — present for ~60% of physics.optics rows). For Nature
Photonics it is the `https://doi.org/<doi>` link (present for all rows).

**Publication-year definition.** For arXiv, "year" = the year of `versions[0].created`
(the first v1 submission), falling back to the 4-digit year of `update_date`. This is
applied identically in both arXiv scripts, so physics.optics counts match between the two
files. For Nature Photonics, "year" = the Crossref published date.

> ⚠️ **2026 is partial** (the snapshot/data was taken mid-2026), so the final year is
> dropped from trend/bump/growth charts and only the years through **2025** are treated as
> complete.

---

## 2. Topic extraction (the core text pipeline)

All title-text analyses (trends, bump charts, rising stars) share **one** extractor,
defined once in the notebook and applied everywhere. A title becomes a set of **topics**
via these steps:

1. **Tokenize** — lowercase; match `[a-z][a-z\-]+`. Hyphens are kept *inside* tokens, so
   `single-photon` and `non-hermitian` are single tokens.
2. **Lemmatize** (`singularize`) — a conservative rule-based singularizer that maps
   plural/singular to **one real word** (`lasers`→`laser`, `photonics`→`photonic`,
   `metasurfaces`→`metasurface`). It has a **protected set** (never singularize:
   `physics`, `optics`, `analysis`, `lens`, …), an **irregulars** map (`indices`→`index`),
   special handling for `-lens`/`-lenses` (so `metalens` isn't mangled to `metalen`), and
   a small **alias** map (`moir`→`moire`, since the letters-only tokenizer truncates the
   accented `moiré`).
3. **Drop stopwords** — a custom set of generic English + scientific-prose filler
   (`using`, `novel`, `study`, `result`, `system`, `reply`/`erratum`, …). Stopwords are
   stored as singular/base forms; the lemmatizer maps plurals onto them.
4. **n-grams** — form adjacent **uni-, bi-, and tri-grams** from the surviving content
   tokens, so multi-word concepts (`frequency comb`, `neural network`,
   `perovskite solar cell`) count as units.

### Canonical keys (order- and hyphen-independent)

Each n-gram is reduced to a **canonical key** = the **sorted set of its constituent words**
(splitting each token on hyphens and re-singularizing). This makes counting independent of
word order *and* hyphenation:

- `machine-learning` and `machine learning` → key `{learning, machine}` → **merged**.
- `photonic integrated` and `integrated photonic` → same key → **merged**.
- `non-hermitian` → key `{hermitian, non}`. Because `non`/`single`/`thin` are stopwords,
  a phrase like `non hermitian` never forms (the stopword is removed), so these
  hyphenated compounds **never collide** with anything and are **preserved** intact. This
  is why we key on the compound rather than stripping hyphens up front (which would
  destroy `non-hermitian`→`hermitian`, `single-photon`→`photon`).

**Readable labels.** Sorting words would make ugly labels (`generation harmonic second`),
so for each canonical key we track the **most common surface form actually seen** and
display that. Counting uses the key; display uses the surface form.

### Manually excluded topics

A few words are frequent but uninformative (`optical`, `light`, …). The notebook
has an **`EXCLUDE_TOPICS`** knob (its own cell, just after the extractor): any
entry is canonicalized the same way titles are (singularize → split hyphens →
sort) and dropped from the topic table, so it disappears from **every** chart and
table at once (trends, bump charts, both rising-star metrics). Editing the set and
re-running rebuilds all downstream figures, letting the next-most-meaningful
topics surface. Matching is on the whole canonical key, so excluding `light`
removes the bare word but **keeps** compound topics like `structured light`.

### Counting, pruning, caching

- **Document frequency** — each topic is counted **once per title** (presence, not raw
  term frequency), tabulated per year. Throughout, a topic's prevalence is reported as a
  **share**: the % of that year's titles containing it. Using a share rather than a raw
  count prevents the steadily rising article volume from distorting trends.
- **Pruning** — canonical keys seen **fewer than 5 times total** are dropped. The long
  trigram tail (mostly singletons) can never reach any top-N or rising-star threshold, so
  this is lossless for the outputs and keeps the term table small.
- **Caching** — the full topic table is computed **once per dataset** (`_full_topic_table`,
  cached by object id); year-windowed views are just column slices. `singularize` and the
  per-title gram builder are memoized. (Naive recomputation took ~100 s; this brings a
  full notebook run to ~45 s.)

---

## 3. Analyses

### Articles per year
Simple `value_counts` of `year` for each dataset, shown as a combined table, separate bar
panels (the two differ ~10× in volume), and a dual-axis overlay.

### Cross-listing categories (arXiv physics.optics)
The `categories` field is exploded; `physics.optics` itself is dropped; the top-10
co-listed categories are tracked per year, both as raw counts and as a **share of that
year's physics.optics papers**.
> Note: `physics.app-ph` and `eess.IV` were created by arXiv in **2017**, so their jump
> from zero is a taxonomy artifact, not a research surge.

### Growth vs. the rest of arXiv
From `arxiv_all.csv`, category membership is matched by **"any-listing"** regex on the
space-delimited category string. Three views over **2000–2025**:
- **Share of all arXiv** — physics.optics articles ÷ total arXiv articles per year (the
  direct "faster/slower than the rest" test).
- **Indexed growth** — each series rebased to **2010 = 100**.
- **CAGR 2010→2025** per field.

### Title-word trends
Top-10 topics by total document frequency, plotted as their **% share of titles per year**
(one chart for each dataset; physics.optics restricted to 2000+ to avoid tiny-sample noise).

### Bump charts (rank over time)
- The **20 most frequent topics of the most recent full year** (2025) are taken, then
  ranked **among themselves** each year, so every line stays in-band and wanders smoothly.
- **Strictly unique ranks**: if two topics tie on count, the tie is broken by their
  **previous year's rank** — no two lines ever share a spot.
- Smoothstep easing (flat-at-the-nodes → S-curve between years) for the bump look.
- Every line gets a **distinct pastel-rainbow colour**, *interleaved* by final rank
  (warm/cool alternating) so neighbouring lines contrast; every line is labelled.
- The term selection is run through the phrase de-duplicator (below).

### Rising-star topics (two complementary metrics)

Both metrics are defined **once** as shared functions (`decade_risers`,
`recent_momentum_risers`) and run on **both** datasets — arXiv physics.optics *and*
Nature Photonics — with identical logic; only the volume floors are scaled, since
Nature Photonics publishes ~20× fewer (and shorter) titles per year.

**(a) Decade comparison** — what's grown big since ~10 years ago.
- recent = 2021–2025, baseline = 2011–2015.
- Keep topics with a recent/baseline share **ratio ≥ 2** and real recent presence:
  **≥ 80** recent titles for physics.optics, **≥ 15** for Nature Photonics.
- Rank by the **absolute rise in share** (percentage points). Shown as a then-vs-now
  dumbbell plus yearly trajectories.

**(b) Recent momentum (last 5 years)** — what's *emerging* now, even if still small.
- Window 2020–2025 (5 year-on-year steps).
- Rank by **relative growth rate** = trend slope ÷ the topic's own mean share, so a topic
  climbing 0.05%→0.45% beats one drifting 8%→9%.
- **Anti-fluke filters**: **sustained** growth (share rose in **≥ 4 of the 5** steps),
  plus a volume floor (physics.optics: **≥ 60** titles in the window, **≥ 8** in 2025;
  Nature Photonics: **≥ 15** and **≥ 3**). A one-year spike cannot qualify. Shown as a
  relative-growth bar + yearly trajectories.

For physics.optics the decade risers are metasurface, topological, non-Hermitian,
integrated/lithium-niobate photonics, ML/neural networks; the recent-momentum list
is optical skyrmions, photonic processors, large-scale integration, metalenses. For
Nature Photonics the decade risers are perovskite / perovskite solar cells,
light-emitting diodes, organic & integrated photonics and metasurfaces, with solar
cells, organic LEDs and quantum leading the recent-momentum list.

### Phrase de-duplication (`dedup_phrases`)
Used to choose which topics to display in the rising-star lists and the bump charts. A
term is dropped if it is a **contiguous sub-phrase of a longer candidate whose count is
nearly as large** (ratio ≥ 0.6) — i.e. they co-occur. This keeps the most specific phrase
(`bound state continuum` instead of `bound state` + `state continuum`; `neural network`
instead of `neural` + `network`) while **keeping an independently-frequent shorter term**
(e.g. `solar cell` survives even when `perovskite solar cell` is present, because it's far
more frequent on its own).

---

## 4. Browsing the raw titles (`browse_titles.ipynb`)

A separate, lightweight notebook for **manual** look-ups (no aggregation). After a
one-cell setup it offers three helpers over both title masters:

- `find(words, source=…, match=…, since=…, until=…, whole_word=…, newest_first=…, limit=…)`
  — search titles for a word, a list of words (`match="all"`/`"any"`), or an exact
  phrase; filter by year; sort by date. `source` is `"optics"`, `"nphoton"` or
  `"both"`. Returns a normal DataFrame for further slicing.
- `show(df)` — render the matches as a table with **each title linked to its
  article** (arXiv abstract page or journal DOI).
- `print_links(df)` / `per_year(df)` — copy-pasteable year + title + URL lines, and
  a per-year match count.

This is the practical use of the new `url`/`doi` columns: go from a topic of
interest straight to the papers.

## 5. Known limitations

- **2026 is partial** and is excluded from trend/bump/growth charts.
- **Tokenizer is letters-only**, so non-ASCII and numerals are stripped: `MoS₂` → `mos`,
  `moiré` → `moir` (re-aliased to `moire`). Niche topics may show slightly truncated
  labels.
- **arXiv category taxonomy evolves** (new categories appear over time), which inflates
  apparent growth of recently-created categories.
- **Cross-listed counting** — a physics.optics paper is counted if it lists the category
  *anywhere*, not only as primary; "any-listing" group membership in the growth section
  likewise double-counts papers across groups (fine for trend comparison, not a partition).
- **Lemmatization is rule-based**, not a full morphological analyzer; a few forms split or
  merge imperfectly (e.g. `learning` and `machine learning` can both appear when `learning`
  is independently frequent across many ML phrases).

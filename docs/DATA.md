# Data and artifact policy

The public repository intentionally excludes benchmark questions, generated
nearby answers, web-search results, full agent trajectories, model caches, and
credentials. This boundary prevents accidental benchmark redistribution,
public-web contamination, and disclosure of third-party content.

`lazy-grounding validate-data` checks a locally prepared manifest without
printing question text. See [the manifest schema](MANIFEST.md) and
`examples/toy/`.

Official sources:

- GAIA: <https://huggingface.co/datasets/gaia-benchmark/GAIA>
- XBench: <https://xbench.org/> and <https://arxiv.org/abs/2506.13651>
- BrowseComp+: <https://github.com/texttron/BrowseComp-Plus>
- Humanity's Last Exam: <https://github.com/centerforaisafety/hle>

Do not assume that one repository license covers separately hosted benchmark
files. Review the dataset card, access agreement, and current upstream terms at
download time. Record the accepted terms and upstream revision locally.

For each private experimental dataset, record:

- benchmark name, release, split, and upstream URL;
- upstream license or terms of use;
- selection seed and sample size;
- stable local item identifiers;
- rewrite generator and verifier versions;
- annotation status; and
- a SHA-256 digest of the manifest used for each run.

Do not upload private manifests as CI fixtures. Use synthetic records under
`examples/toy/` for tests and documentation.

# Third-party software and data

The Apache-2.0 license in this repository covers only original Lazy Grounding
code. It does not relicense models, benchmarks, web content, or third-party
software.

## Agent scaffold

The experiments use a text ReAct scaffold adapted from Tongyi DeepResearch:

- Project: `Alibaba-NLP/DeepResearch`
- URL: <https://github.com/Alibaba-NLP/DeepResearch>
- Model: `Alibaba-NLP/Tongyi-DeepResearch-30B-A3B`
- License: Apache-2.0

Python execution can use SandboxFusion:

- Project: `bytedance/SandboxFusion`
- URL: <https://github.com/bytedance/SandboxFusion>
- Tested revision: `5e37f71a5f61bd7dddd1fa867b5cb7be01a1bbb6`
- License: Apache-2.0

These projects are installed separately and are not vendored in this release.

## Benchmarks

GAIA, XBench DeepSearch, BrowseComp+, and Humanity's Last Exam remain subject to
their original licenses and access terms. This repository provides loaders and
configuration, not redistributed benchmark questions. Users are responsible for
obtaining each dataset from its official source and complying with its terms.

The evaluated hosted models, EmbeddingGemma weights, Serper, Jina Reader, and
provider APIs are also governed by their respective licenses and service terms.
Dependency versions used by this package are pinned in `uv.lock`; model and
service versions are recorded in the resolved experiment config.

## Search results

Search snippets and visited pages may be copyrighted or contain personal data.
Raw traces are therefore excluded from the source release. Inspect and redact
artifacts before sharing them.

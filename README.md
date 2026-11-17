# hayashiyus.github.io

Reviewed static research blog, published with GitHub Pages from `main` at `/`.

- [All articles / 記事一覧](https://hayashiyus.github.io/blog/)
- [What Did the Model Preserve? — English research proposal](https://hayashiyus.github.io/blog/codebook-ontology/en/)
- [English article](https://hayashiyus.github.io/blog/weight-space-geometry/en/)
- [日本語版](https://hayashiyus.github.io/blog/weight-space-geometry/ja/)
- [Source archive and reproduction materials](blog/weight-space-geometry/en/downloads/research-feature.zip)

The archive contains the editable article sources, build scripts, selected media,
measurement aggregates, model revisions and reproduction instructions. Its README
distinguishes completed probes, mathematical examples and unperformed experiments.
Model weights and benchmark data are obtained separately. The E5 file is a planning
snapshot, not a runnable experiment package. Third-party material retains its own
licenses and attribution; see the archive and article for provenance.

Only current reviewed English and Japanese builds are staged here. Local review
pages, connection scripts, credentials and server-specific information are excluded.
`publication.json` records hashes of the reviewed manuscripts and figure manifests.

## Codebook / ontology article

`blog/codebook-ontology/en/` contains the reviewed 12 November 2026 English article, watercolor animation, figures, local KaTeX runtime/fonts, source Markdown, and finite/algebra verification scripts. The VAE theorem, separate finite construction, and proposed (unperformed) open-weight LLM experiments are distinguished throughout. `blog/codebook-ontology/publication.json` records release hashes.

## Blog URLs and future articles

`/blog/` is the article index. Publish new articles below `/blog/<article-slug>/<language>/` and add their links and original publication dates to `blog/index.html` and `sitemap.xml`. Existing article URLs are permanent, including the English and Japanese WSP links already shared on X. Adding an article to the index does not change its URL, publication date, or content.

This repository maintains the site-wide index, homepage, sitemap, and README. When copying a new article build from a separate source project, update only that article's subtree; do not replace these shared files with a single-article staging build.

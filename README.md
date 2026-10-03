# japanese-rag-swift (JapaneseRAGKit)

[English](#english) | [日本語](#日本語)

## English

A small Swift package for **on-device Japanese RAG** (retrieval-augmented generation) on iOS / macOS: Japanese sentence chunking,
[ruri-v3](https://huggingface.co/cl-nagoya/ruri-v3-130m) embeddings and a Japanese cross-encoder reranker on Core ML, an in-memory vector index
and a retrieve -> rerank pipeline. It does not include an LLM; use it to pick the passages you give to Apple's Foundation Models, MLX, llama.cpp, etc.

| Piece | What it does |
|---|---|
| `JapaneseChunker` | splits on 。！？ and newlines, packs sentences up to a character limit, with sentence overlap |
| `CoreMLEmbedder` + `PrefixedSentenceEmbedder` | ruri-v3 Core ML model -> L2-normalized embedding; applies ruri's `検索クエリ: ` / `検索文書: ` prefixes |
| `CoreMLReranker` | cross-encoder reranker (`[CLS] query [SEP] document [SEP]`, sigmoid score) |
| `VectorIndex` | exact in-memory search (dot product), replace-by-id, JSON save / load |
| `RAGPipeline` | `ingest(documentID:text:)` and `search(query, retrieveK:, topN:)`, with or without a reranker |

```swift
import JapaneseRAGKit

let base = try await CoreMLEmbedder(compiledModelURL: modelURL, tokenizerFolder: tokenizerURL, sequenceLength: 128)
let rag = RAGPipeline(
    queryEmbedder: PrefixedSentenceEmbedder(base, prefix: .searchQuery),
    documentEmbedder: PrefixedSentenceEmbedder(base, prefix: .searchDocument),
    reranker: nil)                                   // or a CoreMLReranker
try await rag.ingest(documentID: "tower", text: "東京タワーは港区にある電波塔で、高さは333メートルです。")
let hits = try await rag.search("港区にある電波塔", retrieveK: 20, topN: 5)
```

**Models and tokenizer.** Core ML models: [masahiroid/ruri-v3-130m-coreml](https://huggingface.co/masahiroid/ruri-v3-130m-coreml) /
[310m](https://huggingface.co/masahiroid/ruri-v3-310m-coreml) (compile `.mlpackage` with Xcode, or `xcrun coremlcompiler compile`).
Use the tokenizer in `tokenizers/ruri-v3/` (swift-transformers cannot load the base repo's tokenizer config as is). Prefer **int8 or fp32**
on real iPhones; fp16 has returned NaN there (the embedder throws a clear error).

**Verified**: 7 unit tests (chunker, index, pipeline with mock embedder/reranker; no model files needed) pass with `swift test`. An end-to-end check on macOS with the
published `ruri-v3-130m` int8 Core ML model (`swift run embed-check`, needs the model in `.models/`) ranks all three sample queries correctly
(e.g. 「日本で一番高い山は？」 -> the Mt. Fuji passage first). **Not yet verified**: the `CoreMLReranker` against a real reranker model (no published Core ML reranker
export is wired in; the pair encoding is implemented but untested on a model), and on-device iPhone timing.

Requires Xcode 16+ / Swift 5.10+, iOS 17+ / macOS 14+. Dependency: [swift-transformers](https://github.com/huggingface/swift-transformers) (tokenizers).

License: Apache-2.0 (matching ruri-v3). Part of [japanese-llm-security](https://github.com/masahirocom/japanese-llm-security)'s sibling work on on-device Japanese AI; code derives from [ruri-coreml](https://github.com/masahirocom/ruri-coreml).

## 日本語

iOS / macOS での**オンデバイス日本語RAG**用の小さなSwiftパッケージです。日本語の文チャンク分割、[ruri-v3](https://huggingface.co/cl-nagoya/ruri-v3-130m)
の埋め込みと日本語クロスエンコーダのリランカー（Core ML）、メモリ上のベクトル索引、検索→再ランキングのパイプラインを含みます。LLMは含みません。
Apple の Foundation Models、MLX、llama.cpp などに渡す文章を選ぶ部分として使います。

| 部品 | 役割 |
|---|---|
| `JapaneseChunker` | 。！？と改行で分割し、文字数上限まで文をまとめ、文の重なりを持たせる |
| `CoreMLEmbedder` + `PrefixedSentenceEmbedder` | ruri-v3のCore MLモデル → L2正規化済みの埋め込み。ruriの `検索クエリ: ` / `検索文書: ` 接頭辞を付与 |
| `CoreMLReranker` | クロスエンコーダのリランカー（`[CLS] クエリ [SEP] 文書 [SEP]`、sigmoidのスコア） |
| `VectorIndex` | メモリ上の厳密検索（内積）、ID単位の置換、JSONの保存・読込 |
| `RAGPipeline` | `ingest(documentID:text:)` と `search(query, retrieveK:, topN:)`（リランカーの有無どちらも可） |

使い方は上の English セクションのコードを参照してください。

**モデルとトークナイザー**: Core MLモデルは [masahiroid/ruri-v3-130m-coreml](https://huggingface.co/masahiroid/ruri-v3-130m-coreml) /
[310m](https://huggingface.co/masahiroid/ruri-v3-310m-coreml)（`.mlpackage` をXcodeまたは `xcrun coremlcompiler compile` でコンパイル）。
トークナイザーは `tokenizers/ruri-v3/` を使ってください（swift-transformersは元リポジトリのトークナイザー設定をそのまま読めません）。実機のiPhoneでは
**int8またはfp32**を推奨します（fp16はNaNを返すことがあり、その場合は明確なエラーを投げます）。

**検証済み**: `swift test` でユニットテスト7件（チャンク、索引、モックの埋め込み・リランカーでのパイプライン。モデルファイル不要）が通ります。公開済みの
`ruri-v3-130m` int8 のCore MLモデルを使ったmacOS上のエンドツーエンド確認（`swift run embed-check`、モデルを `.models/` に置く）で、3つのサンプルクエリが全て
正しく順位付けされました（例: 「日本で一番高い山は？」→ 富士山の文章が1位）。**未検証**: 実際のリランカーモデルでの `CoreMLReranker`（ペアのエンコードは実装済みですが、
モデルでは未確認）と、iPhone実機での速度。

Xcode 16+ / Swift 5.10+、iOS 17+ / macOS 14+。依存: [swift-transformers](https://github.com/huggingface/swift-transformers)（トークナイザー）。

ライセンス: Apache-2.0（ruri-v3に合わせています）。コードは [ruri-coreml](https://github.com/masahirocom/ruri-coreml) から派生しています。

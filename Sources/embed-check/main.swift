import Foundation
import JapaneseRAGKit

// Real-model check: embeds a query and three documents with the published ruri-v3-130m Core ML model and ranks them.
let root = URL(fileURLWithPath: ".models")
let model = root.appendingPathComponent("ruri-v3-130m_seq128_int8.mlmodelc")
let tok = root.appendingPathComponent("tokenizer")
let base = try await CoreMLEmbedder(compiledModelURL: model, tokenizerFolder: tok, sequenceLength: 128)
let pipe = RAGPipeline(queryEmbedder: PrefixedSentenceEmbedder(base, prefix: .searchQuery),
                       documentEmbedder: PrefixedSentenceEmbedder(base, prefix: .searchDocument))
try await pipe.ingest(documentID: "tower", text: "東京タワーは港区にある電波塔で、高さは333メートルです。")
try await pipe.ingest(documentID: "ruri", text: "瑠璃色は、紫みを帯びた濃い青のことである。")
try await pipe.ingest(documentID: "fuji", text: "富士山は静岡県と山梨県にまたがる日本一高い山です。")
for q in ["日本で一番高い山は？", "紫がかった青い色", "港区にある電波塔"] {
    let r = try await pipe.search(q, retrieveK: 3, topN: 3)
    print(q, "->", r.map { "\($0.chunk.id.split(separator: "#")[0]):\(String(format: "%.3f", $0.retrievalScore))" }.joined(separator: "  "))
}

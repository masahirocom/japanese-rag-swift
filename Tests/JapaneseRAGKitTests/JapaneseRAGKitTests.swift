import XCTest
@testable import JapaneseRAGKit

/// Deterministic bag-of-characters embedder: no model files needed.
struct CharEmbedder: SentenceEmbedder {
    func embed(_ text: String) async throws -> [Float] {
        var v = [Float](repeating: 0, count: 64)
        for s in text.unicodeScalars { v[Int(s.value) % 64] += 1 }
        return VectorMath.l2Normalized(v)
    }
}

struct KeywordReranker: Reranker {
    let keyword: String
    func score(query: String, document: String) async throws -> Float { document.contains(keyword) ? 1 : 0 }
}

final class JapaneseRAGKitTests: XCTestCase {
    func testVectorMath() {
        XCTAssertEqual(VectorMath.l2Normalized([3, 4]), [0.6, 0.8])
        XCTAssertEqual(VectorMath.l2Normalized([0, 0]), [0, 0])
        XCTAssertEqual(VectorMath.dotProduct([1, 0], [0.5, 0.5]), 0.5)
        XCTAssertEqual(VectorMath.sigmoid(0), 0.5, accuracy: 1e-6)
    }

    func testPrefixes() {
        XCTAssertEqual(RuriPromptPrefix.searchQuery.applied(to: "天気"), "検索クエリ: 天気")
        XCTAssertEqual(RuriPromptPrefix.searchDocument.applied(to: "晴れ"), "検索文書: 晴れ")
    }

    func testSentenceSplitting() {
        let s = JapaneseChunker().sentences("今日は晴れです。明日は雨？\nそうですね！以上")
        XCTAssertEqual(s, ["今日は晴れです。", "明日は雨？", "そうですね！", "以上"])
    }

    func testChunkerRespectsLimitAndOverlap() {
        let text = "一つ目の文です。二つ目の文です。三つ目の文です。四つ目の文です。"
        let chunks = JapaneseChunker(maxCharacters: 18, overlapSentences: 1).chunk(text)
        XCTAssertTrue(chunks.allSatisfy { $0.count <= 18 })
        XCTAssertGreaterThan(chunks.count, 1)
        XCTAssertTrue(chunks[1].hasPrefix("二つ目の文です。"))   // overlap: last sentence of chunk 0 repeats
        XCTAssertEqual(JapaneseChunker(maxCharacters: 5, overlapSentences: 0).chunk("あいうえおかきく"), ["あいうえお", "かきく"])
    }

    func testIndexSearchReplaceAndDimension() throws {
        var idx = VectorIndex()
        try idx.add(IndexedChunk(id: "a", text: "A", vector: [1, 0]))
        try idx.add(IndexedChunk(id: "b", text: "B", vector: [0, 1]))
        XCTAssertEqual(try idx.search([1, 0], topK: 1).first?.chunk.id, "a")
        try idx.add(IndexedChunk(id: "a", text: "A2", vector: [0, 1]))   // same id replaces
        XCTAssertEqual(idx.count, 2)
        XCTAssertThrowsError(try idx.add(IndexedChunk(id: "c", text: "C", vector: [1, 0, 0])))
        XCTAssertThrowsError(try idx.search([1, 0, 0], topK: 1))
    }

    func testIndexPersistence() throws {
        var idx = VectorIndex()
        try idx.add(IndexedChunk(id: "a", text: "あ", vector: [1, 0], metadata: ["src": "x"]))
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("idx-\(UUID().uuidString).json")
        try idx.save(to: url)
        let back = try VectorIndex.load(from: url)
        XCTAssertEqual(back.chunks, idx.chunks)
        try? FileManager.default.removeItem(at: url)
    }

    func testPipelineRetrievesAndReranks() async throws {
        let e = CharEmbedder()
        let p = RAGPipeline(
            queryEmbedder: PrefixedSentenceEmbedder(e, prefix: .none), documentEmbedder: PrefixedSentenceEmbedder(e, prefix: .none),
            reranker: KeywordReranker(keyword: "東京タワー"), chunker: JapaneseChunker(maxCharacters: 40)
        )
        try await p.ingest(documentID: "d1", text: "東京タワーは港区にある電波塔です。高さは333メートルです。")
        try await p.ingest(documentID: "d2", text: "瑠璃色は紫みを帯びた濃い青のことです。")
        let r = try await p.search("東京タワーはどこにある？", retrieveK: 5, topN: 2)
        XCTAssertEqual(r.first?.chunk.id.hasPrefix("d1"), true)
        XCTAssertEqual(r.first?.rerankScore, 1)
        let plain = RAGPipeline(queryEmbedder: e, documentEmbedder: e)
        try await plain.ingest(documentID: "d2", text: "瑠璃色は紫みを帯びた濃い青のことです。")
        let r2 = try await plain.search("瑠璃色", topN: 1)
        XCTAssertNil(r2.first?.rerankScore)
    }
}

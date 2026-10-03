import Foundation

public struct RankedPassage: Sendable, Equatable {
    public let chunk: IndexedChunk
    public let retrievalScore: Float
    public let rerankScore: Float?
}

/// Retrieval (+ optional reranking) over an in-memory index:
/// ingest text -> chunk -> embed (document prefix) -> index; query -> embed (query prefix) -> top-K -> rerank -> top-N.
public actor RAGPipeline {
    private let queryEmbedder: any SentenceEmbedder
    private let documentEmbedder: any SentenceEmbedder
    private let reranker: (any Reranker)?
    private let chunker: JapaneseChunker
    public private(set) var index: VectorIndex

    /// Pass one ruri embedder wrapped twice with `PrefixedSentenceEmbedder` (`.searchQuery` / `.searchDocument`).
    public init(
        queryEmbedder: any SentenceEmbedder,
        documentEmbedder: any SentenceEmbedder,
        reranker: (any Reranker)? = nil,
        chunker: JapaneseChunker = JapaneseChunker(),
        index: VectorIndex = VectorIndex()
    ) {
        self.queryEmbedder = queryEmbedder
        self.documentEmbedder = documentEmbedder
        self.reranker = reranker
        self.chunker = chunker
        self.index = index
    }

    /// Chunks `text`, embeds each chunk and adds it to the index. Returns the number of chunks added.
    @discardableResult
    public func ingest(documentID: String, text: String, metadata: [String: String] = [:]) async throws -> Int {
        let pieces = chunker.chunk(text)
        for (i, piece) in pieces.enumerated() {
            let v = VectorMath.l2Normalized(try await documentEmbedder.embed(piece))
            try index.add(IndexedChunk(id: "\(documentID)#\(i)", text: piece, vector: v, metadata: metadata))
        }
        return pieces.count
    }

    /// Finds `retrieveK` candidates by embedding similarity; with a reranker, rescored and cut to `topN`.
    public func search(_ query: String, retrieveK: Int = 20, topN: Int = 5) async throws -> [RankedPassage] {
        let q = VectorMath.l2Normalized(try await queryEmbedder.embed(query))
        let hits = try index.search(q, topK: retrieveK)
        guard let reranker else {
            return hits.prefix(topN).map { RankedPassage(chunk: $0.chunk, retrievalScore: $0.score, rerankScore: nil) }
        }
        var rescored: [RankedPassage] = []
        for hit in hits {
            let s = try await reranker.score(query: query, document: hit.chunk.text)
            rescored.append(RankedPassage(chunk: hit.chunk, retrievalScore: hit.score, rerankScore: s))
        }
        return Array(rescored.sorted { ($0.rerankScore ?? 0) > ($1.rerankScore ?? 0) }.prefix(topN))
    }
}

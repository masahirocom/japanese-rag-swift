import Foundation

public struct IndexedChunk: Codable, Sendable, Equatable {
    public let id: String
    public let text: String
    public var metadata: [String: String]
    public var vector: [Float]

    public init(id: String, text: String, vector: [Float], metadata: [String: String] = [:]) {
        self.id = id; self.text = text; self.vector = vector; self.metadata = metadata
    }
}

public struct SearchHit: Sendable, Equatable {
    public let chunk: IndexedChunk
    public let score: Float
}

/// A small in-memory vector index (exact search by dot product; vectors are expected to be L2-normalized).
/// Good for thousands of chunks on a phone; persist with `save(to:)` / `load(from:)`.
public struct VectorIndex: Codable, Sendable {
    public private(set) var chunks: [IndexedChunk] = []
    public private(set) var dimension: Int?

    public init() {}
    public var count: Int { chunks.count }

    public mutating func add(_ chunk: IndexedChunk) throws {
        if let d = dimension, d != chunk.vector.count { throw RAGError.dimensionMismatch(expected: d, got: chunk.vector.count) }
        dimension = chunk.vector.count
        if let i = chunks.firstIndex(where: { $0.id == chunk.id }) { chunks[i] = chunk } else { chunks.append(chunk) }
    }

    public mutating func remove(id: String) { chunks.removeAll { $0.id == id } }

    public func search(_ query: [Float], topK: Int) throws -> [SearchHit] {
        if let d = dimension, d != query.count { throw RAGError.dimensionMismatch(expected: d, got: query.count) }
        let scored = chunks.map { SearchHit(chunk: $0, score: VectorMath.dotProduct(query, $0.vector)) }
        return Array(scored.sorted { $0.score > $1.score }.prefix(max(topK, 0)))
    }

    public func save(to url: URL) throws { try JSONEncoder().encode(self).write(to: url, options: .atomic) }
    public static func load(from url: URL) throws -> VectorIndex { try JSONDecoder().decode(VectorIndex.self, from: Data(contentsOf: url)) }
}

import Foundation

/// Turns text into a (usually L2-normalized) embedding vector.
public protocol SentenceEmbedder: Sendable {
    func embed(_ text: String) async throws -> [Float]
}

/// ruri-v3 was trained with a prefix scheme ("1+3"): the same encoder works better when the text is prefixed with its role.
/// See https://huggingface.co/cl-nagoya/ruri-v3-130m
public enum RuriPromptPrefix: String, Sendable {
    case none = ""
    case topic = "トピック: "
    case searchQuery = "検索クエリ: "
    case searchDocument = "検索文書: "

    public func applied(to text: String) -> String { rawValue + text }
}

/// Applies a fixed prefix before delegating, so one model can serve as both the query and the document embedder.
public struct PrefixedSentenceEmbedder: SentenceEmbedder {
    private let base: any SentenceEmbedder
    private let prefix: RuriPromptPrefix

    public init(_ base: any SentenceEmbedder, prefix: RuriPromptPrefix) {
        self.base = base
        self.prefix = prefix
    }

    public func embed(_ text: String) async throws -> [Float] {
        try await base.embed(prefix.applied(to: text))
    }
}

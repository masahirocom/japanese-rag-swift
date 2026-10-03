import Foundation

/// Splits Japanese text into chunks for embedding: sentence boundaries (。！？!? and newlines) are kept, sentences are packed up
/// to `maxCharacters`, and `overlapSentences` trailing sentences are repeated at the start of the next chunk.
/// Characters, not tokens: ruri-v3 models take 128-512 tokens and Japanese averages roughly 1-1.5 characters per token.
public struct JapaneseChunker: Sendable {
    public let maxCharacters: Int
    public let overlapSentences: Int

    public init(maxCharacters: Int = 200, overlapSentences: Int = 1) {
        self.maxCharacters = max(maxCharacters, 1)
        self.overlapSentences = max(overlapSentences, 0)
    }

    public func sentences(_ text: String) -> [String] {
        var out: [String] = []
        var current = ""
        let terminators: Set<Character> = ["。", "！", "？", "!", "?", "\n"]
        for ch in text {
            current.append(ch)
            if terminators.contains(ch) {
                let s = current.trimmingCharacters(in: .whitespacesAndNewlines)
                if !s.isEmpty { out.append(s) }
                current = ""
            }
        }
        let tail = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if !tail.isEmpty { out.append(tail) }
        return out
    }

    public func chunk(_ text: String) -> [String] {
        // a sentence longer than the limit is cut into pieces first
        var units: [String] = []
        for s in sentences(text) {
            var rest = Substring(s)
            while rest.count > maxCharacters {
                units.append(String(rest.prefix(maxCharacters)))
                rest = rest.dropFirst(maxCharacters)
            }
            if !rest.isEmpty { units.append(String(rest)) }
        }
        var chunks: [String] = []
        var window: [String] = []
        var length = 0
        for u in units {
            if length + u.count > maxCharacters, !window.isEmpty {
                chunks.append(window.joined())
                let keep = overlapSentences > 0 ? Array(window.suffix(overlapSentences)) : []
                window = keep.filter { $0.count + u.count <= maxCharacters }
                length = window.reduce(0) { $0 + $1.count }
            }
            window.append(u)
            length += u.count
        }
        if !window.isEmpty { chunks.append(window.joined()) }
        return chunks
    }
}

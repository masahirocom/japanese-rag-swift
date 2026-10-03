import Foundation

/// Pure vector-math helpers (no model or framework knowledge).
public enum VectorMath {
    /// For unit vectors the dot product is the cosine similarity.
    public static func dotProduct(_ lhs: [Float], _ rhs: [Float]) -> Float {
        precondition(lhs.count == rhs.count, "Vectors must have the same dimensionality to compare.")
        var sum: Float = 0
        for index in lhs.indices { sum += lhs[index] * rhs[index] }
        return sum
    }

    /// Scales `vector` to unit length; an all-zero vector is returned unchanged.
    public static func l2Normalized(_ vector: [Float]) -> [Float] {
        let magnitude = sqrt(vector.reduce(Float(0)) { $0 + $1 * $1 })
        guard magnitude > .leastNormalMagnitude else { return vector }
        return vector.map { $0 / magnitude }
    }

    /// 1 / (1 + e^-x), used to turn a reranker logit into a 0-1 score.
    public static func sigmoid(_ x: Float) -> Float { 1 / (1 + exp(-x)) }
}

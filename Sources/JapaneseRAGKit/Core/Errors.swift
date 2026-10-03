import Foundation

public enum RAGError: Error, CustomStringConvertible, Equatable {
    case missingOutputFeature(String)
    case nanOutput(count: Int, dimension: Int)
    case invalidTokenizer(String)
    case dimensionMismatch(expected: Int, got: Int)

    public var description: String {
        switch self {
        case .missingOutputFeature(let n): return "Model has no output feature named '\(n)'."
        case .nanOutput(let c, let d):
            return "Model returned NaN in \(c)/\(d) values. fp16 models can do this on real iPhones; use an fp32 or int8 variant."
        case .invalidTokenizer(let m): return "Tokenizer problem: \(m)"
        case .dimensionMismatch(let e, let g): return "Vector dimension \(g) does not match the index dimension \(e)."
        }
    }
}

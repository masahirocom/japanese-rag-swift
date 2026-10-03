import CoreML
import Foundation
import Tokenizers

/// Scores how well a document answers a query (higher is better).
public protocol Reranker: Sendable {
    func score(query: String, document: String) async throws -> Float
}

/// Runs a converted Japanese cross-encoder reranker (Core ML, `input_ids` / `attention_mask` -> relevance logit).
/// The pair is encoded as `[CLS] query [SEP] document [SEP]`, truncating the document first.
public actor CoreMLReranker: Reranker {
    private let model: MLModel
    private let tokenizer: any Tokenizer
    private let sequenceLength: Int
    private let outputFeatureName: String
    private let applySigmoid: Bool

    public init(
        compiledModelURL: URL,
        tokenizerFolder: URL,
        sequenceLength: Int = 256,
        outputFeatureName: String = "logits",
        applySigmoid: Bool = true,
        computeUnits: MLComputeUnits = .all
    ) async throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        self.model = try MLModel(contentsOf: compiledModelURL, configuration: cfg)
        self.tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerFolder)
        self.sequenceLength = sequenceLength
        self.outputFeatureName = outputFeatureName
        self.applySigmoid = applySigmoid
    }

    public func score(query: String, document: String) throws -> Float {
        let q = tokenizer.encode(text: query)                              // [CLS] q [SEP]
        guard let sep = q.last else { throw RAGError.invalidTokenizer("empty encoding") }
        let d = tokenizer.encode(text: document, addSpecialTokens: false)  // document tokens only
        let room = max(sequenceLength - q.count - 1, 0)
        let ids = q + Array(d.prefix(room)) + [sep]
        let enc = FixedLengthEncoding(tokenIDs: ids, sequenceLength: sequenceLength)
        let out = try model.prediction(from: enc.featureProvider())
        guard let array = out.featureValue(for: outputFeatureName)?.multiArrayValue else {
            throw RAGError.missingOutputFeature(outputFeatureName)
        }
        let logit = array.jrk_floatVector().first ?? .nan
        if logit.isNaN { throw RAGError.nanOutput(count: 1, dimension: 1) }
        return applySigmoid ? VectorMath.sigmoid(logit) : logit
    }
}

import CoreML
import Foundation
import Tokenizers

/// Runs a converted ruri-v3 Core ML model (`input_ids` / `attention_mask` -> mean-pooled, L2-normalized embedding).
/// Models: https://huggingface.co/masahiroid (ruri-v3-*-coreml). The tokenizer folder is the base model's
/// (`tokenizer.json`, `tokenizer_config.json`).
public actor CoreMLEmbedder: SentenceEmbedder {
    private let model: MLModel
    private let tokenizer: any Tokenizer
    private let sequenceLength: Int
    private let outputFeatureName: String

    public init(
        compiledModelURL: URL,
        tokenizerFolder: URL,
        sequenceLength: Int = 128,
        outputFeatureName: String = "sentence_embedding",
        computeUnits: MLComputeUnits = .all
    ) async throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        self.model = try MLModel(contentsOf: compiledModelURL, configuration: cfg)
        self.tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerFolder)
        self.sequenceLength = sequenceLength
        self.outputFeatureName = outputFeatureName
    }

    public func embed(_ text: String) throws -> [Float] {
        let enc = FixedLengthEncoding(tokenIDs: tokenizer.encode(text: text), sequenceLength: sequenceLength)
        let out = try model.prediction(from: enc.featureProvider())
        guard let array = out.featureValue(for: outputFeatureName)?.multiArrayValue else {
            throw RAGError.missingOutputFeature(outputFeatureName)
        }
        let vector = array.jrk_floatVector()
        let nan = vector.filter(\.isNaN).count
        if nan > 0 { throw RAGError.nanOutput(count: nan, dimension: vector.count) }
        return vector
    }
}

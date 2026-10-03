import CoreML
import Foundation

extension MLMultiArray {
    /// Copies the contents out as `[Float]`. float16 arrays are read through their native byte layout:
    /// `NSNumber.floatValue` on a float16 array has returned NaN on real devices while working in the Simulator.
    func jrk_floatVector() -> [Float] {
        guard dataType == .float16 else { return (0..<count).map { self[$0].floatValue } }
        let p = dataPointer.bindMemory(to: Float16.self, capacity: count)
        return (0..<count).map { Float(p[$0]) }
    }
}

/// Token ids + attention mask padded/truncated to a fixed length (the shape the converted graphs expect).
struct FixedLengthEncoding {
    let tokenIDs: [Int]
    let attentionMask: [Int]

    init(tokenIDs: [Int], sequenceLength: Int, padID: Int = 0) {
        let t = tokenIDs.count > sequenceLength ? Array(tokenIDs.prefix(sequenceLength)) : tokenIDs
        let pad = sequenceLength - t.count
        self.tokenIDs = t + Array(repeating: padID, count: pad)
        self.attentionMask = Array(repeating: 1, count: t.count) + Array(repeating: 0, count: pad)
    }

    func featureProvider() throws -> MLFeatureProvider {
        let shape: [NSNumber] = [1, NSNumber(value: tokenIDs.count)]
        let ids = try MLMultiArray(shape: shape, dataType: .int32)
        let mask = try MLMultiArray(shape: shape, dataType: .int32)
        for i in 0..<tokenIDs.count {
            ids[i] = NSNumber(value: tokenIDs[i])
            mask[i] = NSNumber(value: attentionMask[i])
        }
        return try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: ids),
            "attention_mask": MLFeatureValue(multiArray: mask),
        ])
    }
}

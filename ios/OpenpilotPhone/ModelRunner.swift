import Foundation
import CoreML

// Model geometry. Defined here in Swift (not imported from the C macros) on purpose.
enum Dims {
    static let imgC = 6
    static let imgH = 128
    static let imgW = 256
    static let imgLen = imgC * imgH * imgW          // one camera image
    static let inputLen = 2 * imgLen                // new_img: narrow + wide
    static let outLen = 2576
    static let idxN = 33
}

enum ModelError: Error, CustomStringConvertible {
    case modelNotFound
    case badArray(String)
    case missingOutput(String)

    var description: String {
        switch self {
        case .modelNotFound: return "driving_supercombo.mlmodelc not found in app bundle"
        case .badArray(let s): return "bad MLMultiArray: \(s)"
        case .missingOutput(let s): return "missing model output: \(s)"
        }
    }
}

enum ComputeChoice: String, CaseIterable, Identifiable {
    case all = "All"
    case cpuOnly = "CPU only"
    case cpuAndGPU = "CPU and GPU"
    case cpuAndNE = "CPU and Neural Engine"

    var id: String { rawValue }

    var units: MLComputeUnits {
        switch self {
        case .all: return .all
        case .cpuOnly: return .cpuOnly
        case .cpuAndGPU: return .cpuAndGPU
        case .cpuAndNE: return .cpuAndNeuralEngine
        }
    }
}

/// IEEE half (bit pattern) to Float, without using the Float16 type (not available on all simulator archs).
@inline(__always)
func halfBitsToFloat(_ h: UInt16) -> Float {
    let sign: UInt32 = UInt32(h & 0x8000) << 16
    let exp: Int = Int((h >> 10) & 0x1F)
    let man: UInt32 = UInt32(h & 0x3FF)
    var bits: UInt32
    if exp == 0 {
        if man == 0 {
            bits = sign
        } else {
            // subnormal half: normalise
            var e: Int = -1
            var m: UInt32 = man
            repeat {
                e += 1
                m <<= 1
            } while (m & 0x400) == 0
            let newExp: UInt32 = UInt32(112 - e)
            bits = sign | (newExp << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 0x1F {
        bits = sign | 0x7F800000 | (man << 13)
    } else {
        let newExp: UInt32 = UInt32(exp + 112)
        bits = sign | (newExp << 23) | (man << 13)
    }
    return Float(bitPattern: bits)
}

enum MultiArrayUtil {
    static func ints(_ n: [NSNumber]) -> [Int] { return n.map { $0.intValue } }

    static func isContiguous(_ a: MLMultiArray) -> Bool {
        let shape = ints(a.shape)
        let strides = ints(a.strides)
        var expected = 1
        var d = shape.count - 1
        while d >= 0 {
            if shape[d] != 1 && strides[d] != expected { return false }
            expected *= shape[d]
            d -= 1
        }
        return true
    }

    /// Zero-filled contiguous float32 array.
    static func makeZeroFloat32(shape: [Int]) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        guard isContiguous(a) else { throw ModelError.badArray("new float32 array is not contiguous") }
        memset(a.dataPointer, 0, a.count * MemoryLayout<Float>.size)
        return a
    }

    /// Copies any float16/float32 MLMultiArray (respecting strides) into a dense Float buffer of `count` elements.
    static func copyToFloats(_ src: MLMultiArray, into dst: UnsafeMutablePointer<Float>, count: Int) throws {
        guard src.count == count else {
            throw ModelError.badArray("count \(src.count) != expected \(count)")
        }
        let shape = ints(src.shape)
        let strides = ints(src.strides)
        let contiguous = isContiguous(src)
        let raw = UnsafeRawPointer(src.dataPointer)

        // Reads element at storage offset (in elements).
        func run(_ read: (Int) -> Float) {
            if contiguous {
                for i in 0..<count { dst[i] = read(i) }
            } else {
                var idx = [Int](repeating: 0, count: shape.count)
                for i in 0..<count {
                    var off = 0
                    for d in 0..<shape.count { off += idx[d] * strides[d] }
                    dst[i] = read(off)
                    var d = shape.count - 1
                    while d >= 0 {
                        idx[d] += 1
                        if idx[d] < shape[d] { break }
                        idx[d] = 0
                        d -= 1
                    }
                }
            }
        }

        switch src.dataType {
        case .float32:
            let p = raw.assumingMemoryBound(to: Float.self)
            run { p[$0] }
        case .float16:
            let p = raw.assumingMemoryBound(to: UInt16.self)
            if contiguous {
                for i in 0..<count { dst[i] = halfBitsToFloat(p[i]) }
            } else {
                run { halfBitsToFloat(p[$0]) }
            }
        default:
            throw ModelError.badArray("unsupported data type \(src.dataType.rawValue)")
        }
    }

    static func floats(_ src: MLMultiArray) throws -> [Float] {
        let n = src.count
        var out = [Float](repeating: 0, count: n)
        try out.withUnsafeMutableBufferPointer { bp in
            try copyToFloats(src, into: bp.baseAddress!, count: n)
        }
        return out
    }
}

/// Owns the Core ML model and the three recurrent state arrays. Not thread safe: use from one queue.
final class ModelRunner {
    static let latActionT: Float = 0.2
    static let longActionT: Float = 0.5

    let computeUnits: MLComputeUnits
    private let model: MLModel

    private let newImg: MLMultiArray
    private let desire: MLMultiArray
    private let trafficConvention: MLMultiArray
    private let actionT: MLMultiArray
    private let stateImg: MLMultiArray
    private let stateDesire: MLMultiArray
    private let stateFeat: MLMultiArray

    private static let stateImgCount = 2 * 5 * 6 * 128 * 256
    private static let stateDesireCount = 100 * 1 * 8
    private static let stateFeatCount = 96 * 1 * 512

    init(computeUnits: MLComputeUnits) throws {
        guard let url = Bundle.main.url(forResource: "driving_supercombo", withExtension: "mlmodelc") else {
            throw ModelError.modelNotFound
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        self.model = try MLModel(contentsOf: url, configuration: cfg)
        self.computeUnits = computeUnits

        newImg = try MultiArrayUtil.makeZeroFloat32(shape: [2, 6, 128, 256])
        desire = try MultiArrayUtil.makeZeroFloat32(shape: [8])
        trafficConvention = try MultiArrayUtil.makeZeroFloat32(shape: [1, 2])
        actionT = try MultiArrayUtil.makeZeroFloat32(shape: [1, 2])
        stateImg = try MultiArrayUtil.makeZeroFloat32(shape: [2, 5, 6, 128, 256])
        stateDesire = try MultiArrayUtil.makeZeroFloat32(shape: [100, 1, 8])
        stateFeat = try MultiArrayUtil.makeZeroFloat32(shape: [96, 1, 512])

        // Constants for the prototype: traffic_convention = [1,0], action_t = [0.2,0.5], desire = 0.
        let tc = trafficConvention.dataPointer.assumingMemoryBound(to: Float.self)
        tc[0] = 1; tc[1] = 0
        let at = actionT.dataPointer.assumingMemoryBound(to: Float.self)
        at[0] = ModelRunner.latActionT; at[1] = ModelRunner.longActionT
    }

    /// Back to the all-zero state (what the first run expects).
    func resetState() {
        memset(stateImg.dataPointer, 0, ModelRunner.stateImgCount * MemoryLayout<Float>.size)
        memset(stateDesire.dataPointer, 0, ModelRunner.stateDesireCount * MemoryLayout<Float>.size)
        memset(stateFeat.dataPointer, 0, ModelRunner.stateFeatCount * MemoryLayout<Float>.size)
    }

    /// One model step. `image` holds 2 x OP_IMG_LEN floats (values 0..255), layout [2][6][128][256].
    /// Returns the 2576 outputs and the wall-clock milliseconds spent inside the prediction call.
    /// `desireIndex`, when set, pulses that entry of the 8-long desire input to 1 for this step only.
    func step(image: [Float], desireIndex: Int? = nil) throws -> (outputs: [Float], ms: Double) {
        guard image.count == Dims.inputLen else {
            throw ModelError.badArray("image has \(image.count) floats, expected \(Dims.inputLen)")
        }
        image.withUnsafeBufferPointer { bp in
            memcpy(newImg.dataPointer, bp.baseAddress!, Dims.inputLen * MemoryLayout<Float>.size)
        }

        let dp = desire.dataPointer.assumingMemoryBound(to: Float.self)
        for k in 0..<8 { dp[k] = 0 }
        if let di = desireIndex, di >= 0, di < 8 { dp[di] = 1 }

        let inputs: [String: Any] = [
            "new_img": MLFeatureValue(multiArray: newImg),
            "desire": MLFeatureValue(multiArray: desire),
            "traffic_convention": MLFeatureValue(multiArray: trafficConvention),
            "action_t": MLFeatureValue(multiArray: actionT),
            "state_img_q": MLFeatureValue(multiArray: stateImg),
            "state_desire_q": MLFeatureValue(multiArray: stateDesire),
            "state_feat_q": MLFeatureValue(multiArray: stateFeat),
        ]
        let provider = try MLDictionaryFeatureProvider(dictionary: inputs)

        let t0 = ProcessInfo.processInfo.systemUptime
        let result = try model.prediction(from: provider)
        let ms = (ProcessInfo.processInfo.systemUptime - t0) * 1000.0

        guard let outArr = result.featureValue(for: "outputs")?.multiArrayValue else {
            throw ModelError.missingOutput("outputs")
        }
        let outputs = try MultiArrayUtil.floats(outArr)
        guard outputs.count == Dims.outLen else {
            throw ModelError.badArray("outputs has \(outputs.count) values, expected \(Dims.outLen)")
        }

        // Carry state: next_state_* (float16) -> state_* (float32) for the next run.
        guard let nImg = result.featureValue(for: "next_state_img_q")?.multiArrayValue else {
            throw ModelError.missingOutput("next_state_img_q")
        }
        guard let nDes = result.featureValue(for: "next_state_desire_q")?.multiArrayValue else {
            throw ModelError.missingOutput("next_state_desire_q")
        }
        guard let nFeat = result.featureValue(for: "next_state_feat_q")?.multiArrayValue else {
            throw ModelError.missingOutput("next_state_feat_q")
        }
        try MultiArrayUtil.copyToFloats(nImg,
                                        into: stateImg.dataPointer.assumingMemoryBound(to: Float.self),
                                        count: ModelRunner.stateImgCount)
        try MultiArrayUtil.copyToFloats(nDes,
                                        into: stateDesire.dataPointer.assumingMemoryBound(to: Float.self),
                                        count: ModelRunner.stateDesireCount)
        try MultiArrayUtil.copyToFloats(nFeat,
                                        into: stateFeat.dataPointer.assumingMemoryBound(to: Float.self),
                                        count: ModelRunner.stateFeatCount)
        return (outputs, ms)
    }
}

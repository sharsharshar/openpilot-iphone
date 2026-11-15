import Foundation
import CoreML

// Model geometry. Defined here in Swift (not imported from the C macros) on purpose.
enum Dims {
    static let imgC = 6
    static let imgH = 128
    static let imgW = 256
    static let imgLen = imgC * imgH * imgW          // one camera image
    static let inputLen = 2 * imgLen                // one step of camera images: narrow + wide
    static let histLen = 5                          // images kept per camera by the frame history
    static let histBytes = histLen * inputLen       // uint8 storage of the whole history
    static let stackLen = 2 * 2 * imgLen            // img_stack [1,24,128,256]: per camera, image 4 steps ago + newest
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

/// Monotonic clock in seconds (does not jump with wall-clock changes), for the per-stage timings.
enum Mono {
    static func now() -> Double { return ProcessInfo.processInfo.systemUptime }
    static func ms(since t: Double) -> Double { return (now() - t) * 1000.0 }
}

/// The last 5 model images per camera, kept as uint8 by the C ring buffer in opcore.c. Builds the lean model's
/// img_stack input from it. Before 5 pushes the missing images read as zeros (as the old float queue did).
final class FrameHistory {
    private let mem: UnsafeMutablePointer<UInt8>
    private var hist = op_hist_t()

    init() {
        mem = UnsafeMutablePointer<UInt8>.allocate(capacity: Dims.histBytes)
        op_hist_init(&hist, mem)
    }

    deinit { mem.deallocate() }

    func reset() { op_hist_reset(&hist) }

    /// One step of images, Dims.inputLen bytes, layout [2][6][128][256].
    func push(bytes: UnsafePointer<UInt8>) { op_hist_push_u8(&hist, bytes) }

    /// One step of images as floats holding integer values 0..255 (the warp output), Dims.inputLen floats.
    func push(floats: UnsafePointer<Float>) { op_hist_push_f32(&hist, floats) }

    /// Writes Dims.stackLen floats (raw 0..255 values) in the layout the model's img_stack input expects.
    func fillStack(_ dst: UnsafeMutablePointer<Float>) { op_hist_fill_stack(&hist, dst) }
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

/// Owns the Core ML model, the frame history and the two small recurrent state arrays. Not thread safe: use from one queue.
final class ModelRunner {
    static let latActionT: Float = 0.2
    static let longActionT: Float = 0.5

    let computeUnits: MLComputeUnits
    private let model: MLModel

    private let history = FrameHistory()
    private let imgStack: MLMultiArray
    private let desire: MLMultiArray
    private let trafficConvention: MLMultiArray
    private let actionT: MLMultiArray
    private let stateDesire: MLMultiArray
    private let stateFeat: MLMultiArray

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

        imgStack = try MultiArrayUtil.makeZeroFloat32(shape: [1, 24, 128, 256])
        desire = try MultiArrayUtil.makeZeroFloat32(shape: [8])
        trafficConvention = try MultiArrayUtil.makeZeroFloat32(shape: [1, 2])
        actionT = try MultiArrayUtil.makeZeroFloat32(shape: [1, 2])
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
        history.reset()
        memset(stateDesire.dataPointer, 0, ModelRunner.stateDesireCount * MemoryLayout<Float>.size)
        memset(stateFeat.dataPointer, 0, ModelRunner.stateFeatCount * MemoryLayout<Float>.size)
    }

    /// One model step. `push` is called first and must append this step's images to the history it is given
    /// (Dims.inputLen values, layout [2][6][128][256], 0..255). The runner then builds img_stack straight into the
    /// model's input array. Returns the 2576 outputs and, in milliseconds on a monotonic clock: `ms`, the prediction
    /// call only; `buildMs`, history push + img_stack + input arrays; `carryMs`, reading outputs and carrying state.
    /// `desireIndex`, when set, pulses that entry of the 8-long desire input to 1 for this step only.
    func step(desireIndex: Int? = nil, push: (FrameHistory) -> Void) throws
        -> (outputs: [Float], ms: Double, buildMs: Double, carryMs: Double) {
        let b0 = Mono.now()
        push(history)
        history.fillStack(imgStack.dataPointer.assumingMemoryBound(to: Float.self))

        let dp = desire.dataPointer.assumingMemoryBound(to: Float.self)
        for k in 0..<8 { dp[k] = 0 }
        if let di = desireIndex, di >= 0, di < 8 { dp[di] = 1 }

        let inputs: [String: Any] = [
            "img_stack": MLFeatureValue(multiArray: imgStack),
            "desire": MLFeatureValue(multiArray: desire),
            "traffic_convention": MLFeatureValue(multiArray: trafficConvention),
            "action_t": MLFeatureValue(multiArray: actionT),
            "state_desire_q": MLFeatureValue(multiArray: stateDesire),
            "state_feat_q": MLFeatureValue(multiArray: stateFeat),
        ]
        let provider = try MLDictionaryFeatureProvider(dictionary: inputs)
        let buildMs = Mono.ms(since: b0)

        let t0 = Mono.now()
        let result = try model.prediction(from: provider)
        let ms = Mono.ms(since: t0)

        let c0 = Mono.now()

        guard let outArr = result.featureValue(for: "outputs")?.multiArrayValue else {
            throw ModelError.missingOutput("outputs")
        }
        let outputs = try MultiArrayUtil.floats(outArr)
        guard outputs.count == Dims.outLen else {
            throw ModelError.badArray("outputs has \(outputs.count) values, expected \(Dims.outLen)")
        }

        // Carry state: next_state_* (float16) -> state_* (float32) for the next run.
        guard let nDes = result.featureValue(for: "next_state_desire_q")?.multiArrayValue else {
            throw ModelError.missingOutput("next_state_desire_q")
        }
        guard let nFeat = result.featureValue(for: "next_state_feat_q")?.multiArrayValue else {
            throw ModelError.missingOutput("next_state_feat_q")
        }
        try MultiArrayUtil.copyToFloats(nDes,
                                        into: stateDesire.dataPointer.assumingMemoryBound(to: Float.self),
                                        count: ModelRunner.stateDesireCount)
        try MultiArrayUtil.copyToFloats(nFeat,
                                        into: stateFeat.dataPointer.assumingMemoryBound(to: Float.self),
                                        count: ModelRunner.stateFeatCount)
        return (outputs, ms, buildMs, Mono.ms(since: c0))
    }
}

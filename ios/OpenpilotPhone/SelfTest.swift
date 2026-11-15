import Foundation
import CoreML
import os

/// The 40 recorded steps bundled with the app. 40, not fewer: a dropped feature queue only shows up after
/// step 8, and a dropped desire queue only after the desire pulse at step 20.
struct Fixtures {
    static let steps = 40
    static let desireStep = 20     // the reference run pulses desire[3] at step 20
    static let desireIndex = 3
    let frames: Data          // steps x [2,6,128,256] uint8
    let expected: [Float]     // steps x 2576 float32

    static func load() throws -> Fixtures {
        guard let fu = Bundle.main.url(forResource: "frames_u8", withExtension: "bin"),
              let ou = Bundle.main.url(forResource: "outputs_f32", withExtension: "bin") else {
            throw NSError(domain: "Fixtures", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "fixture .bin files not found in bundle"])
        }
        let frames = try Data(contentsOf: fu)
        let outData = try Data(contentsOf: ou)
        guard frames.count == steps * Dims.inputLen else {
            throw NSError(domain: "Fixtures", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "frames_u8.bin has \(frames.count) bytes, expected \(steps * Dims.inputLen)"])
        }
        guard outData.count == steps * Dims.outLen * 4 else {
            throw NSError(domain: "Fixtures", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "outputs_f32.bin has \(outData.count) bytes, expected \(steps * Dims.outLen * 4)"])
        }
        var expected = [Float](repeating: 0, count: steps * Dims.outLen)
        expected.withUnsafeMutableBufferPointer { dst in
            _ = outData.copyBytes(to: dst)
        }
        return Fixtures(frames: frames, expected: expected)
    }

    /// Frame `i` as 2 x OP_IMG_LEN floats.
    func frameFloats(_ i: Int) -> [Float] {
        let start = i * Dims.inputLen
        var out = [Float](repeating: 0, count: Dims.inputLen)
        frames.withUnsafeBytes { raw in
            let p = raw.bindMemory(to: UInt8.self)
            for k in 0..<Dims.inputLen {
                out[k] = Float(p[start + k])
            }
        }
        return out
    }

    func expectedOutputs(_ i: Int) -> [Float] {
        let start = i * Dims.outLen
        return Array(expected[start..<(start + Dims.outLen)])
    }
}

enum SelfTest {
    static let logger = Logger(subsystem: "com.example.openpilotphone", category: "selftest")

    static func documentsURL() -> URL? {
        return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
    }

    /// Blocking; call from a background queue. Returns (passed, full text).
    static func run(units: MLComputeUnits) -> (passed: Bool, text: String) {
        var lines: [String] = []
        func emit(_ s: String) {
            lines.append(s)
            print(s)
            NSLog("%@", s)
            logger.notice("\(s, privacy: .public)")
        }

        var passed = true
        do {
            emit("SELFTEST start compute_units \(units.rawValue)")
            let fixtures = try Fixtures.load()
            let runner = try ModelRunner(computeUnits: units)
            for i in 0..<Fixtures.steps {
                let r = try runner.step(image: fixtures.frameFloats(i),
                                        desireIndex: i == Fixtures.desireStep ? Fixtures.desireIndex : nil)
                let exp = fixtures.expectedOutputs(i)
                var maxD: Double = 0
                var sumD: Double = 0
                for k in 0..<Dims.outLen {
                    let d = Double(abs(r.outputs[k] - exp[k]))
                    if d.isNaN { maxD = Double.infinity; sumD = Double.infinity; break }
                    if d > maxD { maxD = d }
                    sumD += d
                }
                let meanD = sumD / Double(Dims.outLen)
                let ok = meanD < 0.05 && maxD < 3.0
                if !ok { passed = false }
                emit(String(format: "SELFTEST step %ld max_abs_diff %.5f mean_abs_diff %.5f ms %.1f",
                            i, maxD, meanD, r.ms))
            }
        } catch {
            passed = false
            emit("SELFTEST error \(error)")
        }
        emit(passed ? "SELFTEST PASS" : "SELFTEST FAIL")
        fflush(stdout)

        let text = lines.joined(separator: "\n") + "\n"
        if let dir = documentsURL() {
            let url = dir.appendingPathComponent("selftest.txt")
            do {
                try text.write(to: url, atomically: true, encoding: .utf8)
            } catch {
                NSLog("SELFTEST could not write selftest.txt: %@", "\(error)")
            }
        }
        return (passed, text)
    }
}

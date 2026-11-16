import Foundation
import CoreML
import os

/// One bundled set of recorded steps and the outputs the float32 reference model gave for them.
struct FixtureSet {
    let tag: String           // log prefix
    let framesName: String
    let outputsName: String
    let desireStep: Int?      // step at which the reference run pulsed desire[Fixtures.desireIndex], if any
    let vEgo: Float           // m/s, used only to turn the plan into curvature / acceleration

    /// Synthetic frames. Checks the plumbing: a dropped feature queue only shows up after step 8, and a
    /// dropped desire queue only after the desire pulse at step 20.
    static let synthetic = FixtureSet(tag: "SELFTEST", framesName: "frames_u8", outputsName: "outputs_f32",
                                      desireStep: 20, vEgo: 15.0)
    /// Real road frames (see fixtures/README.md), no desire pulse. Speed is the car's logged average
    /// over these frames. Shows what a numeric difference does to the driving outputs.
    static let real = FixtureSet(tag: "REALTEST", framesName: "real_frames_u8", outputsName: "real_outputs_f32",
                                 desireStep: nil, vEgo: 17.0)
}

/// 40 recorded steps bundled with the app.
struct Fixtures {
    static let steps = 40
    static let desireIndex = 3
    let frames: Data          // steps x [2,6,128,256] uint8
    let expected: [Float]     // steps x 2576 float32

    static func load(_ set: FixtureSet = .synthetic) throws -> Fixtures {
        guard let fu = Bundle.main.url(forResource: set.framesName, withExtension: "bin"),
              let ou = Bundle.main.url(forResource: set.outputsName, withExtension: "bin") else {
            throw NSError(domain: "Fixtures", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "\(set.framesName).bin / \(set.outputsName).bin not found in bundle"])
        }
        let frames = try Data(contentsOf: fu)
        let outData = try Data(contentsOf: ou)
        guard frames.count == steps * Dims.inputLen else {
            throw NSError(domain: "Fixtures", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "\(set.framesName).bin has \(frames.count) bytes, expected \(steps * Dims.inputLen)"])
        }
        guard outData.count == steps * Dims.outLen * 4 else {
            throw NSError(domain: "Fixtures", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "\(set.outputsName).bin has \(outData.count) bytes, expected \(steps * Dims.outLen * 4)"])
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

    /// Calls `body` with a pointer to frame `i` as uint8 (Dims.inputLen bytes), straight from the fixture data.
    func withFrameBytes(_ i: Int, _ body: (UnsafePointer<UInt8>) -> Void) {
        frames.withUnsafeBytes { raw in
            let p = raw.bindMemory(to: UInt8.self)
            body(p.baseAddress! + i * Dims.inputLen)
        }
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

    private static func parse(_ outputs: [Float], vEgo: Float, prev: op_result_t?) -> op_result_t {
        var out = op_result_t()
        outputs.withUnsafeBufferPointer { bp in
            if var p = prev {
                op_parse_outputs(bp.baseAddress, vEgo, ModelRunner.latActionT, ModelRunner.longActionT, &p, &out)
            } else {
                op_parse_outputs(bp.baseAddress, vEgo, ModelRunner.latActionT, ModelRunner.longActionT, nil, &out)
            }
        }
        return out
    }

    /// Forward distance of the plan at the last time index (10 s ahead), metres.
    private static func planX10s(_ r: op_result_t) -> Float {
        var pos = r.plan_pos
        return withUnsafeBytes(of: &pos) { $0.bindMemory(to: Float.self)[32 * 3] }
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

        // The verdict line "SELFTEST PASS/FAIL" is for the synthetic set only, as before. The real-footage set
        // reports its own "REALTEST PASS/FAIL" with the same limits, plus the change in the driving outputs.
        var passed = true
        for set in [FixtureSet.synthetic, FixtureSet.real] {
            let tag = set.tag
            var setPassed = true
            var worstCurv: Double = 0, worstAccel: Double = 0, worstPlanX: Double = 0, worstLane: Double = 0
            do {
                emit("\(tag) start compute_units \(units.rawValue)")
                let fixtures = try Fixtures.load(set)
                let runner = try ModelRunner(computeUnits: units)   // fresh model state for each set
                var prevGot = op_result_t(), prevRef = op_result_t()
                var havePrev = false
                for i in 0..<Fixtures.steps {
                    let r = try runner.step(desireIndex: i == set.desireStep ? Fixtures.desireIndex : nil) { history in
                        fixtures.withFrameBytes(i) { history.push(bytes: $0) }
                    }
                    let exp = fixtures.expectedOutputs(i)
                    let p0 = Mono.now()
                    let got = parse(r.outputs, vEgo: set.vEgo, prev: havePrev ? prevGot : nil)
                    let parseMs = Mono.ms(since: p0)
                    let ref = parse(exp, vEgo: set.vEgo, prev: havePrev ? prevRef : nil)
                    prevGot = got; prevRef = ref; havePrev = true

                    var maxD: Double = 0
                    var sumD: Double = 0
                    var maxK = 0
                    var nBig = 0
                    for k in 0..<Dims.outLen {
                        let d = Double(abs(r.outputs[k] - exp[k]))
                        if d.isNaN { maxD = Double.infinity; sumD = Double.infinity; break }
                        if d > maxD { maxD = d; maxK = k }
                        if d > 0.5 { nBig += 1 }
                        sumD += d
                    }
                    let meanD = sumD / Double(Dims.outLen)
                    let ok = meanD < 0.05 && maxD < 3.0
                    if !ok { setPassed = false }

                    // Driving outputs: what the car would be asked to do, and two things a driver would see.
                    let dCurv = Double(abs(got.desired_curvature - ref.desired_curvature))
                    let dAccel = Double(abs(got.desired_accel - ref.desired_accel))
                    let dPlanX = Double(abs(planX10s(got) - planX10s(ref)))
                    let gl = got.lane_line_probs, rl = ref.lane_line_probs
                    let dLane = Double(max(abs(gl.0 - rl.0), abs(gl.1 - rl.1), abs(gl.2 - rl.2), abs(gl.3 - rl.3)))
                    worstCurv = max(worstCurv, dCurv); worstAccel = max(worstAccel, dAccel)
                    worstPlanX = max(worstPlanX, dPlanX); worstLane = max(worstLane, dLane)

                    emit(String(format: "%@ step %ld max_abs_diff %.5f mean_abs_diff %.5f ms %.1f worst_idx %ld ref %.3f got %.3f n_over_0.5 %ld build_ms %.2f predict_ms %.2f carry_ms %.2f parse_ms %.2f curv_ref %.6f curv_diff %.6f accel_ref %.4f accel_diff %.4f planx10_ref %.2f planx10_diff %.3f lane_prob_diff %.4f",
                                tag, i, maxD, meanD, r.ms, maxK, Double(exp[maxK]), Double(r.outputs[maxK]), nBig,
                                r.buildMs, r.ms, r.carryMs, parseMs,
                                Double(ref.desired_curvature), dCurv, Double(ref.desired_accel), dAccel,
                                Double(planX10s(ref)), dPlanX, dLane))
                }
                emit(String(format: "%@ driving worst_curv_diff %.6f worst_accel_diff %.4f worst_planx10_diff %.3f worst_lane_prob_diff %.4f",
                            tag, worstCurv, worstAccel, worstPlanX, worstLane))
            } catch {
                setPassed = false
                emit("\(tag) error \(error)")
            }
            if set.tag == FixtureSet.synthetic.tag {
                passed = setPassed
            } else {
                emit(setPassed ? "\(tag) PASS" : "\(tag) FAIL")
            }
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

import Foundation
import CoreML
import CoreVideo
import Combine

struct Pt {
    var x: Float   // metres forward
    var y: Float   // metres lateral (model convention: positive = right)
}

struct DisplayResult {
    var plan: [Pt] = []
    var laneLines: [[Pt]] = []
    var laneProbs: [Float] = []
    var roadEdges: [[Pt]] = []
    var curvature: Float = 0
    var accel: Float = 0
    var shouldStop: Bool = false
    var leadProb: Float = 0
    var leadX: Float = 0
}

/// Reading fixed-size C arrays (imported into Swift as tuples) as flat Float arrays.
enum CRead {
    static func floats<T>(_ v: inout T, count: Int) -> [Float] {
        return withUnsafeBytes(of: &v) { (raw: UnsafeRawBufferPointer) -> [Float] in
            let p = raw.bindMemory(to: Float.self)
            return Array(p.prefix(count))
        }
    }

    static func floats<T>(copyOf v: T, count: Int) -> [Float] {
        var copy = v
        return floats(&copy, count: count)
    }
}

/// Camera frames -> model inputs -> model -> parsed result. Published properties are only
/// written on the main queue; all heavy work happens on `processQueue`.
final class Pipeline: ObservableObject {
    static let vEgo: Float = 15.0   // m/s, fixed: no car is attached

    @Published var display = DisplayResult()
    @Published var msPerRun: Double = 0
    @Published var fps: Double = 0
    @Published var dropped: Int = 0
    @Published var statusText: String = "idle"
    @Published var sourceName: String = ""
    @Published var computeChoice: ComputeChoice = .all
    @Published var selfTestText: String = ""
    @Published var selfTestRunning: Bool = false
    @Published var aneProbeText: String = ""
    @Published var aneProbeRunning: Bool = false

    private let processQueue = DispatchQueue(label: "pipeline.process", qos: .userInitiated)
    private let lock = NSLock()
    private var busy = false
    private var loading = false
    private var droppedCount = 0

    // Touched only on processQueue:
    private var runner: ModelRunner?
    private var prev = op_result_t()
    private var havePrev = false
    private var liveCount = 0
    private var lastDone: Double = 0
    private var fpsEMA: Double = 0

    // Touched on the main queue:
    private var source: FrameSource?
    private var cameraRef: CameraSource?
    private var launched = false

    // MARK: lifecycle (call on main)

    /// Entry point from the UI. With `-selftest` on the command line, runs only the self-test.
    func launch() {
        if launched { return }
        launched = true
        let args = CommandLine.arguments
        if args.contains("-aneprobe") {
            // ANE probe only: no camera, no model pipeline.
            runANEProbe()
            return
        }
        if args.contains("-selftest") {
            // CPU only by default: deterministic, and CI VMs may have no usable GPU.
            var units = MLComputeUnits.cpuOnly
            if let i = args.firstIndex(of: "-units"), i + 1 < args.count {
                switch args[i + 1] {
                case "all": units = .all
                case "gpu": units = .cpuAndGPU
                case "ne": units = .cpuAndNeuralEngine
                default: units = .cpuOnly
                }
            }
            runSelfTest(units: units, restart: false)
        } else {
            start()
        }
    }

    func start() {
        if source != nil { return }
        let s: FrameSource
        #if targetEnvironment(simulator)
        s = FixtureSource()
        sourceName = "fixtures (simulator has no camera)"
        #else
        let cam = CameraSource()
        cameraRef = cam
        s = cam
        sourceName = "back camera"
        #endif
        source = s
        loadModel(choice: computeChoice) { [weak self] ok in
            guard let self = self, ok else { return }
            if self.source !== s { return }   // stopped while the model was loading
            s.start(handler: { [weak self] frame in
                self?.handle(frame)
            }, status: { [weak self] msg in
                DispatchQueue.main.async { self?.statusText = msg }
            })
        }
    }

    func stop() {
        source?.stop()
        source = nil
        cameraRef = nil
    }

    /// Reload the model with new compute units (state restarts from zeros).
    func reload(choice: ComputeChoice) {
        loadModel(choice: choice) { _ in }
    }

    private func loadModel(choice: ComputeChoice, completion: @escaping (Bool) -> Void) {
        lock.lock()
        loading = true
        lock.unlock()
        statusText = "loading model (\(choice.rawValue))..."
        processQueue.async { [weak self] in
            guard let self = self else { return }
            var ok = false
            var msg = ""
            do {
                let r = try ModelRunner(computeUnits: choice.units)
                self.runner = r
                self.prev = op_result_t()
                self.havePrev = false
                self.lastDone = 0
                self.fpsEMA = 0
                ok = true
                msg = "running"
            } catch {
                self.runner = nil
                msg = "model load failed: \(error)"
            }
            self.lock.lock()
            self.loading = false
            self.lock.unlock()
            DispatchQueue.main.async {
                self.statusText = msg
                completion(ok)
            }
        }
    }

    // MARK: self-test

    func runSelfTest(units: MLComputeUnits? = nil, restart: Bool = true) {
        if selfTestRunning { return }
        selfTestRunning = true
        selfTestText = "running..."
        let wasRunning = (source != nil)
        stop()
        let u = units ?? computeChoice.units
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let result = SelfTest.run(units: u)
            DispatchQueue.main.async {
                guard let self = self else { return }
                self.selfTestText = result.text
                self.selfTestRunning = false
                if restart && wasRunning { self.start() }
            }
        }
    }

    func runANEProbe() {
        if aneProbeRunning { return }
        aneProbeRunning = true
        aneProbeText = "running..."
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let text = ANEProbeRunner.run()
            DispatchQueue.main.async {
                guard let self = self else { return }
                self.aneProbeText = text
                self.aneProbeRunning = false
            }
        }
    }

    // MARK: frame handling

    /// Called on the source's queue. Drops the frame if one is already being processed (never queues).
    private func handle(_ frame: Frame) {
        lock.lock()
        if busy || loading {
            droppedCount += 1
            lock.unlock()
            return
        }
        busy = true
        lock.unlock()
        processQueue.async { [weak self] in
            guard let self = self else { return }
            self.process(frame)
            self.lock.lock()
            self.busy = false
            self.lock.unlock()
        }
    }

    private func process(_ frame: Frame) {
        guard let runner = runner else { return }
        let input: [Float]
        var warpMs: Double? = nil      // only measured for camera frames
        switch frame {
        case .modelInput(let f):
            input = f
        case .camera(let pb):
            let w0 = Mono.now()
            guard let f = warp(pb) else { return }
            warpMs = Mono.ms(since: w0)
            input = f
        }

        do {
            let r = try runner.step { history in
                input.withUnsafeBufferPointer { history.push(floats: $0.baseAddress!) }
            }
            let p0 = Mono.now()
            var out = op_result_t()
            r.outputs.withUnsafeBufferPointer { bp in
                if havePrev {
                    op_parse_outputs(bp.baseAddress, Pipeline.vEgo,
                                     ModelRunner.latActionT, ModelRunner.longActionT,
                                     &prev, &out)
                } else {
                    op_parse_outputs(bp.baseAddress, Pipeline.vEgo,
                                     ModelRunner.latActionT, ModelRunner.longActionT,
                                     nil, &out)
                }
            }
            prev = out
            havePrev = true
            let parseMs = Mono.ms(since: p0)
            let disp = Pipeline.makeDisplay(&out)
            liveCount += 1
            if liveCount % 20 == 0 {
                var line = String(format: "LIVE frames %ld ms %.1f curvature %.6f accel %.4f lead_prob %.3f build_ms %.2f predict_ms %.2f carry_ms %.2f parse_ms %.2f",
                                  liveCount, r.ms, Double(out.desired_curvature), Double(out.desired_accel), Double(out.lead_prob),
                                  r.buildMs, r.ms, r.carryMs, parseMs)
                if let w = warpMs { line += String(format: " warp_ms %.2f", w) }
                NSLog("%@", line)
            }

            let now = ProcessInfo.processInfo.systemUptime
            if lastDone > 0 {
                let dt = now - lastDone
                if dt > 0 {
                    let inst = 1.0 / dt
                    fpsEMA = (fpsEMA == 0) ? inst : (0.9 * fpsEMA + 0.1 * inst)
                }
            }
            lastDone = now
            let fpsNow = fpsEMA
            let ms = r.ms
            lock.lock()
            let dropNow = droppedCount
            lock.unlock()

            DispatchQueue.main.async {
                self.display = disp
                self.msPerRun = ms
                self.fps = fpsNow
                self.dropped = dropNow
            }
        } catch {
            let msg = "step failed: \(error)"
            DispatchQueue.main.async { self.statusText = msg }
        }
    }

    /// NV12 camera frame -> 2 x OP_IMG_LEN floats (the same frame feeds both the narrow and wide input).
    ///
    /// Intrinsics from the field of view of the active capture format (pinhole model, square pixels):
    ///   fx = fy = (width / 2) / tan(hfov / 2),   cx = width / 2,   cy = height / 2
    /// where hfov is AVCaptureDevice.Format.videoFieldOfView (horizontal, degrees) and width/height are the
    /// dimensions of the delivered pixel buffer.
    private func warp(_ pb: CVPixelBuffer) -> [Float]? {
        if CVPixelBufferGetPixelFormatType(pb) != kCVPixelFormatType_420YpCbCr8BiPlanarFullRange { return nil }
        if CVPixelBufferGetPlaneCount(pb) != 2 { return nil }
        let w = CVPixelBufferGetWidth(pb)
        let h = CVPixelBufferGetHeight(pb)

        let fovDeg: Float = cameraRef?.horizontalFOVDegrees ?? 63.0
        let fovRad: Float = fovDeg * Float.pi / 180.0
        let f: Float = (Float(w) / 2.0) / tan(fovRad / 2.0)
        let k: [Float] = [f, 0, Float(w) / 2.0,
                          0, f, Float(h) / 2.0,
                          0, 0, 1]
        let rpy: [Float] = [0, 0, 0]
        var mNarrow = [Float](repeating: 0, count: 9)
        var mWide = [Float](repeating: 0, count: 9)
        op_warp_matrix(k, rpy, 0, &mNarrow)
        op_warp_matrix(k, rpy, 1, &mWide)

        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let yBase = CVPixelBufferGetBaseAddressOfPlane(pb, 0),
              let uvBase = CVPixelBufferGetBaseAddressOfPlane(pb, 1) else { return nil }
        let yStride = CVPixelBufferGetBytesPerRowOfPlane(pb, 0)
        let uvStride = CVPixelBufferGetBytesPerRowOfPlane(pb, 1)
        let yPtr = UnsafePointer<UInt8>(yBase.assumingMemoryBound(to: UInt8.self))
        let uvPtr = UnsafePointer<UInt8>(uvBase.assumingMemoryBound(to: UInt8.self))

        var input = [Float](repeating: 0, count: Dims.inputLen)
        input.withUnsafeMutableBufferPointer { bp in
            let base = bp.baseAddress!
            op_frame_to_model_input(yPtr, Int32(yStride), uvPtr, Int32(uvStride),
                                    Int32(w), Int32(h), mNarrow, base)
            op_frame_to_model_input(yPtr, Int32(yStride), uvPtr, Int32(uvStride),
                                    Int32(w), Int32(h), mWide, base + Dims.imgLen)
        }
        return input
    }

    // MARK: C struct -> Swift

    static func makeDisplay(_ r: inout op_result_t) -> DisplayResult {
        let n = Dims.idxN
        let xs = CRead.floats(copyOf: OP_X_IDXS, count: n)
        let pos = CRead.floats(&r.plan_pos, count: n * 3)
        let lanes = CRead.floats(&r.lane_lines, count: 4 * n * 2)
        let laneProbs = CRead.floats(&r.lane_line_probs, count: 4)
        let edges = CRead.floats(&r.road_edges, count: 2 * n * 2)

        var d = DisplayResult()
        var plan: [Pt] = []
        for i in 0..<n {
            // plan_pos is [x, y, z]; y is positive to the right like the lane lines.
            plan.append(Pt(x: pos[i * 3], y: pos[i * 3 + 1]))
        }
        d.plan = plan

        var lines: [[Pt]] = []
        for l in 0..<4 {
            var pts: [Pt] = []
            for i in 0..<n {
                pts.append(Pt(x: xs[i], y: lanes[(l * n + i) * 2]))
            }
            lines.append(pts)
        }
        d.laneLines = lines
        d.laneProbs = laneProbs

        var eds: [[Pt]] = []
        for e in 0..<2 {
            var pts: [Pt] = []
            for i in 0..<n {
                pts.append(Pt(x: xs[i], y: edges[(e * n + i) * 2]))
            }
            eds.append(pts)
        }
        d.roadEdges = eds

        d.curvature = r.desired_curvature
        d.accel = r.desired_accel
        d.shouldStop = (r.should_stop != 0)
        d.leadProb = r.lead_prob
        d.leadX = r.lead_x
        return d
    }
}

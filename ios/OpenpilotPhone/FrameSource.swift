import Foundation
import AVFoundation
import CoreMedia
import CoreVideo

/// What a source delivers: either a raw camera frame (NV12) that still needs warping,
/// or an already-prepared model input (2 x OP_IMG_LEN floats) from the fixtures.
enum Frame {
    case camera(CVPixelBuffer)
    case modelInput([Float])
}

protocol FrameSource: AnyObject {
    /// `handler` is called on a background queue. `status` reports human-readable problems.
    func start(handler: @escaping (Frame) -> Void, status: @escaping (String) -> Void)
    func stop()
}

// MARK: - Fixtures (simulator)

final class FixtureSource: FrameSource {
    private let queue = DispatchQueue(label: "fixture.source", qos: .userInitiated)
    private var timer: DispatchSourceTimer?
    private var frames: [[Float]] = []
    private var index = 0

    func start(handler: @escaping (Frame) -> Void, status: @escaping (String) -> Void) {
        do {
            let fx = try Fixtures.load()
            var all: [[Float]] = []
            for i in 0..<min(8, Fixtures.steps) { all.append(fx.frameFloats(i)) }   // 8 frames is enough to loop
            frames = all
        } catch {
            status("fixtures: \(error)")
            return
        }
        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now(), repeating: .milliseconds(50))   // 20 Hz
        t.setEventHandler { [weak self] in
            guard let self = self, !self.frames.isEmpty else { return }
            let i = self.index % self.frames.count
            self.index += 1
            handler(.modelInput(self.frames[i]))
        }
        timer = t
        t.resume()
    }

    func stop() {
        timer?.cancel()
        timer = nil
    }
}

// MARK: - Camera (device)

final class CameraSource: NSObject, FrameSource, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "camera.session")
    private let frameQueue = DispatchQueue(label: "camera.frames", qos: .userInitiated)
    private var frameHandler: ((Frame) -> Void)?
    private var statusHandler: ((String) -> Void)?
    private var configured = false

    /// Horizontal field of view of the active format in degrees (set once the session is configured).
    private(set) var horizontalFOVDegrees: Float = 63.0

    func start(handler: @escaping (Frame) -> Void, status: @escaping (String) -> Void) {
        frameHandler = handler
        statusHandler = status
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            sessionQueue.async { self.configureAndStart() }
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                if granted {
                    self.sessionQueue.async { self.configureAndStart() }
                } else {
                    status("camera permission denied")
                }
            }
        default:
            status("camera permission denied or restricted")
        }
    }

    func stop() {
        sessionQueue.async {
            if self.session.isRunning { self.session.stopRunning() }
        }
    }

    private func configureAndStart() {
        if !configured {
            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
                statusHandler?("no back wide camera")
                return
            }
            do {
                let input = try AVCaptureDeviceInput(device: device)
                session.beginConfiguration()
                session.sessionPreset = .hd1280x720
                guard session.canAddInput(input) else {
                    session.commitConfiguration()
                    statusHandler?("cannot add camera input")
                    return
                }
                session.addInput(input)

                let output = AVCaptureVideoDataOutput()
                output.videoSettings = [
                    kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)
                ]
                output.alwaysDiscardsLateVideoFrames = true
                output.setSampleBufferDelegate(self, queue: frameQueue)
                guard session.canAddOutput(output) else {
                    session.commitConfiguration()
                    statusHandler?("cannot add video output")
                    return
                }
                session.addOutput(output)
                session.commitConfiguration()

                // 20 fps, only if the active format supports it (an unsupported duration raises an ObjC exception).
                let supports20 = device.activeFormat.videoSupportedFrameRateRanges.contains {
                    $0.minFrameRate <= 20.0 && 20.0 <= $0.maxFrameRate
                }
                if supports20 {
                    try device.lockForConfiguration()
                    device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 20)
                    device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 20)
                    device.unlockForConfiguration()
                }
                horizontalFOVDegrees = device.activeFormat.videoFieldOfView
                configured = true
            } catch {
                statusHandler?("camera setup failed: \(error)")
                return
            }
        }
        if !session.isRunning { session.startRunning() }
    }

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        frameHandler?(.camera(pixelBuffer))
    }
}

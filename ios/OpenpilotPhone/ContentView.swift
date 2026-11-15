import SwiftUI

struct ContentView: View {
    @StateObject private var pipeline = Pipeline()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                PlotView(result: pipeline.display)
                    .frame(height: 380)
                    .background(Color.black)
                    .clipShape(RoundedRectangle(cornerRadius: 8))

                VStack(alignment: .leading, spacing: 4) {
                    Text(String(format: "%.1f ms per run", pipeline.msPerRun))
                    Text(String(format: "%.1f frames per second (dropped %ld)", pipeline.fps, pipeline.dropped))
                    Text(String(format: "desired curvature %.5f 1/m", pipeline.display.curvature))
                    Text(String(format: "desired accel %.2f m/s^2", pipeline.display.accel)
                         + (pipeline.display.shouldStop ? " (stop)" : ""))
                    Text(String(format: "lead probability %.2f (x %.1f m)", pipeline.display.leadProb, pipeline.display.leadX))
                    Text("source: \(pipeline.sourceName)")
                    Text("status: \(pipeline.statusText)")
                }
                .font(.system(.footnote, design: .monospaced))

                HStack {
                    Text("Compute units")
                    Picker("Compute units", selection: $pipeline.computeChoice) {
                        ForEach(ComputeChoice.allCases) { c in
                            Text(c.rawValue).tag(c)
                        }
                    }
                    .pickerStyle(.menu)
                }
                .onChange(of: pipeline.computeChoice) { _, newValue in
                    pipeline.reload(choice: newValue)
                }

                Button("Run self-test") {
                    pipeline.runSelfTest()
                }
                .buttonStyle(.borderedProminent)
                .disabled(pipeline.selfTestRunning)

                if !pipeline.selfTestText.isEmpty {
                    Text(pipeline.selfTestText)
                        .font(.system(size: 10, design: .monospaced))
                        .textSelection(.enabled)
                }
            }
            .padding(16)
        }
        .onAppear {
            pipeline.launch()
        }
    }
}

struct PlotView: View {
    let result: DisplayResult

    private let rangeMeters: CGFloat = 100

    var body: some View {
        Canvas { ctx, size in
            let scale = size.height / rangeMeters
            // Model frame: x forward, y positive to the right. Screen: forward is up.
            func point(_ p: Pt) -> CGPoint {
                return CGPoint(x: size.width / 2 + CGFloat(p.y) * scale,
                               y: size.height - CGFloat(p.x) * scale)
            }
            func makePath(_ pts: [Pt]) -> Path {
                var acc = Path()
                var first = true
                for p in pts {
                    if p.x < 0 || p.x > Float(rangeMeters) { continue }
                    let cp = point(p)
                    if first {
                        acc.move(to: cp)
                        first = false
                    } else {
                        acc.addLine(to: cp)
                    }
                }
                return acc
            }

            // distance rings every 20 m
            for d in stride(from: 20, through: 100, by: 20) {
                let y = size.height - CGFloat(d) * scale
                var ring = Path()
                ring.move(to: CGPoint(x: 0, y: y))
                ring.addLine(to: CGPoint(x: size.width, y: y))
                ctx.stroke(ring, with: .color(Color.gray.opacity(0.25)), lineWidth: 0.5)
            }

            for edge in result.roadEdges {
                ctx.stroke(makePath(edge), with: .color(Color.red.opacity(0.8)), lineWidth: 2)
            }
            for (i, line) in result.laneLines.enumerated() {
                let prob = i < result.laneProbs.count ? Double(result.laneProbs[i]) : 0
                let o = min(max(prob, 0.05), 1.0)
                ctx.stroke(makePath(line), with: .color(Color.white.opacity(o)), lineWidth: 2)
            }
            ctx.stroke(makePath(result.plan), with: .color(Color.green), lineWidth: 3)

            // ego marker
            var ego = Path()
            ego.move(to: CGPoint(x: size.width / 2, y: size.height - 12))
            ego.addLine(to: CGPoint(x: size.width / 2 - 6, y: size.height))
            ego.addLine(to: CGPoint(x: size.width / 2 + 6, y: size.height))
            ego.closeSubpath()
            ctx.fill(ego, with: .color(Color.blue))
        }
    }
}

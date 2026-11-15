import Foundation

/// Runs the ANE probe (experiments/ane_probe) against the bundled compiled model. Blocking; call from a
/// background queue. Every line goes to print and NSLog, and the full text to Documents/aneprobe.txt.
enum ANEProbeRunner {
    static func run() -> String {
        var text = ""
        if let url = Bundle.main.url(forResource: "driving_supercombo", withExtension: "mlmodelc") {
            text = ANEProbeRun(url.path, "driving_supercombo") { line in
                print(line)
                fflush(stdout)
                NSLog("%@", line)
            }
        } else {
            let a = "ANEPROBE error: driving_supercombo.mlmodelc not found in app bundle"
            let b = "ANEPROBE DONE driving_supercombo compile=0 load=0 eval=skipped"
            for l in [a, b] {
                print(l)
                NSLog("%@", l)
            }
            fflush(stdout)
            text = a + "\n" + b + "\n"
        }
        if let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            do {
                try text.write(to: dir.appendingPathComponent("aneprobe.txt"), atomically: true, encoding: .utf8)
            } catch {
                NSLog("ANEPROBE could not write aneprobe.txt: %@", "\(error)")
            }
        }
        return text
    }
}

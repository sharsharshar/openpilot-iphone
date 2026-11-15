// ANE probe: drives Apple's private AppleNeuralEngine.framework classes directly (no private headers),
// bypassing Core ML's scheduler. Interoperability research on the owner's own devices only; never for
// App Store distribution. Every log line starts with "ANEPROBE".
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

// Stage a: environment and class discovery. Returns the full text.
NSString *ANEProbeEnvironment(void (^ _Nullable logLine)(NSString *line));

// Stages b to d for one compiled Core ML model directory (.mlmodelc). Returns the full text.
NSString *ANEProbeModel(NSString *mlmodelcPath, NSString *label, void (^ _Nullable logLine)(NSString *line));

// Stage a followed by stages b to d. Returns the full text.
NSString *ANEProbeRun(NSString *mlmodelcPath, NSString *label, void (^ _Nullable logLine)(NSString *line));

NS_ASSUME_NONNULL_END

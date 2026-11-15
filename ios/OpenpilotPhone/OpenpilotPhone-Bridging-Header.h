#include "opcore.h"

#import <Foundation/Foundation.h>

// ANE probe experiment, implemented in ../experiments/ane_probe/ane_probe.m.
NS_ASSUME_NONNULL_BEGIN
NSString *ANEProbeRun(NSString *mlmodelcPath, NSString *label, void (^ _Nullable logLine)(NSString *line));
NS_ASSUME_NONNULL_END

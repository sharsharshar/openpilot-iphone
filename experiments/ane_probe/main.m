// macOS command-line wrapper for the ANE probe.
// Usage: ane_probe path/to/a.mlmodelc [path/to/b.mlmodelc ...]
// Runs the environment stage once, then the model stages for every path. Exit code is always 0.
#import <Foundation/Foundation.h>
#import <stdio.h>
#import "ane_probe.h"

int main(int argc, char **argv) {
    @autoreleasepool {
        void (^out)(NSString *) = ^(NSString *line) {
            fputs([line UTF8String], stdout);
            fputc('\n', stdout);
            fflush(stdout);
        };
        @try {
            ANEProbeEnvironment(out);
        } @catch (NSException *ex) {
            out([NSString stringWithFormat:@"ANEPROBE EXCEPTION in environment: %@ %@", ex.name, ex.reason]);
        }
        if (argc < 2) {
            out(@"ANEPROBE usage: ane_probe <model.mlmodelc> [more.mlmodelc ...]");
        }
        for (int i = 1; i < argc; i++) {
            NSString *p = [NSString stringWithUTF8String:argv[i]];
            if (!p) {
                out([NSString stringWithFormat:@"ANEPROBE skipping argument %d (not UTF-8)", i]);
                continue;
            }
            if (![p isAbsolutePath]) {
                p = [[[NSFileManager defaultManager] currentDirectoryPath] stringByAppendingPathComponent:p];
            }
            NSString *label = [[p lastPathComponent] stringByDeletingPathExtension];
            @try {
                ANEProbeModel(p, label, out);
            } @catch (NSException *ex) {
                out([NSString stringWithFormat:@"ANEPROBE EXCEPTION for %@: %@ %@", label, ex.name, ex.reason]);
                out([NSString stringWithFormat:@"ANEPROBE DONE %@ compile=0 load=0 eval=skipped", label]);
            }
        }
    }
    return 0;
}

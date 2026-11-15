// See ane_probe.h. Objective-C, ARC, macOS and iOS. Only NSClassFromString / NSSelectorFromString,
// objc_msgSend casts, NSInvocation (introspection only) and dlopen are used.
#import "ane_probe.h"

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurfaceRef.h>
#import <TargetConditionals.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <sys/sysctl.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static const char *kANEFrameworkPath = "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine";
static const NSUInteger kMinSurfaceBytes = 16384;   // smaller outputs fail silently at evaluate

#pragma mark - logging

@interface ANEPLog : NSObject
@property (nonatomic, strong) NSMutableArray<NSString *> *lines;
@property (nonatomic, copy, nullable) void (^sink)(NSString *);
- (instancetype)initWithSink:(nullable void (^)(NSString *))sink;
- (void)log:(NSString *)fmt, ...;
- (NSString *)text;
@end

@implementation ANEPLog
- (instancetype)initWithSink:(void (^)(NSString *))sink {
    if ((self = [super init])) {
        _lines = [NSMutableArray array];
        _sink = sink;
    }
    return self;
}
- (void)log:(NSString *)fmt, ... {
    va_list ap;
    va_start(ap, fmt);
    NSString *msg = [[NSString alloc] initWithFormat:fmt arguments:ap];
    va_end(ap);
    for (NSString *l in [msg componentsSeparatedByString:@"\n"]) {
        NSString *line = [@"ANEPROBE " stringByAppendingString:l];
        [self.lines addObject:line];
        if (self.sink) {
            self.sink(line);
        }
    }
}
- (NSString *)text {
    return [[self.lines componentsJoinedByString:@"\n"] stringByAppendingString:@"\n"];
}
@end

#pragma mark - helpers

static double ANEPSeconds(uint64_t ticks) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) {
        mach_timebase_info(&tb);
    }
    return (double)ticks * (double)tb.numer / (double)tb.denom / 1e9;
}

static NSString *ANEPTrunc(NSString *s, NSUInteger n) {
    if (s.length <= n) {
        return s;
    }
    return [[s substringToIndex:n] stringByAppendingFormat:@"...(+%lu chars)", (unsigned long)(s.length - n)];
}

static NSString *ANEPErr(NSError *e) {
    if (!e) {
        return @"(no NSError returned)";
    }
    return [NSString stringWithFormat:@"domain=%@ code=%ld desc=%@ full=%@",
            e.domain, (long)e.code, e.localizedDescription, e];
}

static const char *ANEPSkipQual(const char *t) {
    while (t && *t && strchr("rnNoORV", *t)) {
        t++;
    }
    return t;
}

// Calls a zero-argument (or one integer/object argument) method through NSInvocation and returns a text
// rendering of the result. Returns nil when the target does not respond to the selector.
static NSString *ANEPDescribeCall(id target, NSString *selName, NSNumber *arg) {
    SEL s = NSSelectorFromString(selName);
    if (!target || ![target respondsToSelector:s]) {
        return nil;
    }
    NSMethodSignature *sig = [target methodSignatureForSelector:s];
    if (!sig) {
        return @"(no method signature)";
    }
    NSUInteger want = arg ? 3 : 2;
    if (sig.numberOfArguments != want) {
        return [NSString stringWithFormat:@"(skipped: method takes %lu args)", (unsigned long)sig.numberOfArguments - 2];
    }
    NSInvocation *inv = [NSInvocation invocationWithMethodSignature:sig];
    inv.target = target;
    inv.selector = s;
    if (arg) {
        const char *at = ANEPSkipQual([sig getArgumentTypeAtIndex:2]);
        switch (*at) {
            case 'I': case 'i': {
                unsigned int v = arg.unsignedIntValue;
                [inv setArgument:&v atIndex:2];
                break;
            }
            case 'Q': case 'q': case 'L': case 'l': {
                unsigned long long v = arg.unsignedLongLongValue;
                [inv setArgument:&v atIndex:2];
                break;
            }
            case '@': {
                __unsafe_unretained id v = arg;
                [inv setArgument:&v atIndex:2];
                break;
            }
            default:
                return [NSString stringWithFormat:@"(skipped: unsupported arg type %s)", at];
        }
    }
    [inv invoke];
    const char *rt = ANEPSkipQual(sig.methodReturnType);
    uint64_t buf[64];
    memset(buf, 0, sizeof(buf));
    NSUInteger len = sig.methodReturnLength;
    switch (*rt) {
        case 'v':
            return @"(void)";
        case '@': {
            __unsafe_unretained id r = nil;
            [inv getReturnValue:&r];
            if (!r) {
                return @"nil";
            }
            return [NSString stringWithFormat:@"<%@> %@", NSStringFromClass([r class]), ANEPTrunc([r description], 1200)];
        }
        case '#': {
            __unsafe_unretained id r = nil;
            [inv getReturnValue:&r];
            return r ? [NSString stringWithFormat:@"class %@", NSStringFromClass(r)] : @"Nil";
        }
        default:
            break;
    }
    if (len > sizeof(buf)) {
        return [NSString stringWithFormat:@"(return value of %lu bytes, type %s)", (unsigned long)len, rt];
    }
    [inv getReturnValue:buf];
    switch (*rt) {
        case 'B': return *(BOOL *)buf ? @"YES" : @"NO";
        case 'c': return [NSString stringWithFormat:@"%d", (int)*(signed char *)buf];
        case 'C': return [NSString stringWithFormat:@"%u", (unsigned)*(unsigned char *)buf];
        case 's': return [NSString stringWithFormat:@"%d", (int)*(short *)buf];
        case 'S': return [NSString stringWithFormat:@"%u", (unsigned)*(unsigned short *)buf];
        case 'i': case 'l': return [NSString stringWithFormat:@"%d", *(int *)buf];
        case 'I': case 'L': return [NSString stringWithFormat:@"%u", *(unsigned int *)buf];
        case 'q': return [NSString stringWithFormat:@"%lld", *(long long *)buf];
        case 'Q': return [NSString stringWithFormat:@"%llu", *(unsigned long long *)buf];
        case 'f': return [NSString stringWithFormat:@"%g", (double)*(float *)buf];
        case 'd': return [NSString stringWithFormat:@"%g", *(double *)buf];
        case '^': return [NSString stringWithFormat:@"pointer %p", *(void **)buf];
        default: return [NSString stringWithFormat:@"(unhandled return type %s, %lu bytes)", rt, (unsigned long)len];
    }
}

static NSString *ANEPMethodNames(Class cls, BOOL meta, NSString *filterRegex, NSUInteger maxChars) {
    if (!cls) {
        return @"(no class)";
    }
    Class target = meta ? object_getClass(cls) : cls;
    unsigned int n = 0;
    Method *ms = class_copyMethodList(target, &n);
    NSRegularExpression *re = filterRegex ? [NSRegularExpression regularExpressionWithPattern:filterRegex
                                                                                      options:NSRegularExpressionCaseInsensitive
                                                                                        error:nil] : nil;
    NSMutableArray<NSString *> *names = [NSMutableArray array];
    for (unsigned int i = 0; i < n; i++) {
        NSString *name = NSStringFromSelector(method_getName(ms[i]));
        if (re && [re numberOfMatchesInString:name options:0 range:NSMakeRange(0, name.length)] == 0) {
            continue;
        }
        [names addObject:name];
    }
    free(ms);
    [names sortUsingSelector:@selector(compare:)];
    return [NSString stringWithFormat:@"%lu of %u: %@", (unsigned long)names.count, n,
            ANEPTrunc([names componentsJoinedByString:@" "], maxChars)];
}

static NSString *ANEPSysctl(const char *name) {
    size_t sz = 0;
    if (sysctlbyname(name, NULL, &sz, NULL, 0) != 0 || sz == 0 || sz > 256) {
        return @"?";
    }
    char buf[257];
    memset(buf, 0, sizeof(buf));
    if (sysctlbyname(name, buf, &sz, NULL, 0) != 0) {
        return @"?";
    }
    return [NSString stringWithUTF8String:buf] ?: @"?";
}

#pragma mark - half precision

static uint16_t ANEPFloatToHalf(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t fe = (x >> 23) & 0xFFu;
    uint32_t m = x & 0x7FFFFFu;
    if (fe == 0xFFu) {
        return (uint16_t)(sign | 0x7C00u | (m ? 0x200u : 0u));
    }
    int32_t e = (int32_t)fe - 127 + 15;
    if (e >= 31) {
        return (uint16_t)(sign | 0x7C00u);
    }
    if (e <= 0) {
        if (e < -10) {
            return (uint16_t)sign;
        }
        m |= 0x800000u;
        int sh = 14 - e;
        uint32_t r = m >> sh;
        if ((m >> (sh - 1)) & 1u) {
            r++;
        }
        return (uint16_t)(sign | r);
    }
    uint32_t r = sign | ((uint32_t)e << 10) | (m >> 13);
    if (m & 0x1000u) {
        r++;
    }
    return (uint16_t)r;
}

static float ANEPHalfToFloat(uint16_t h) {
    uint32_t s = ((uint32_t)h & 0x8000u) << 16;
    uint32_t e = ((uint32_t)h >> 10) & 0x1Fu;
    uint32_t m = (uint32_t)h & 0x3FFu;
    uint32_t x;
    if (e == 0) {
        if (m == 0) {
            x = s;
        } else {
            float f = ldexpf((float)m, -24);
            return s ? -f : f;
        }
    } else if (e == 31) {
        x = s | 0x7F800000u | (m << 13);
    } else {
        x = s | ((e + 112u) << 23) | (m << 13);
    }
    float f;
    memcpy(&f, &x, 4);
    return f;
}

#pragma mark - IOSurface

static IOSurfaceRef ANEPMakeSurface(NSUInteger bytes) {
    if (bytes < kMinSurfaceBytes) {
        bytes = kMinSurfaceBytes;
    }
    NSDictionary *props = @{
        (__bridge id)kIOSurfaceWidth: @(bytes),
        (__bridge id)kIOSurfaceHeight: @1,
        (__bridge id)kIOSurfaceBytesPerElement: @1,
        (__bridge id)kIOSurfaceBytesPerRow: @(bytes),
        (__bridge id)kIOSurfaceAllocSize: @(bytes),
        (__bridge id)kIOSurfacePixelFormat: @0,
    };
    return IOSurfaceCreate((__bridge CFDictionaryRef)props);   // +1, caller CFReleases
}

#pragma mark - MIL signature parsing

// Parsed tensor: dtype name, shape, element count, byte width (0 = unsupported).
@interface ANEPTensor : NSObject
@property (nonatomic, copy) NSString *name;
@property (nonatomic, copy) NSString *dtype;
@property (nonatomic, copy) NSString *shapeText;
@property (nonatomic) NSUInteger elements;
@property (nonatomic) NSUInteger elemBytes;
@end
@implementation ANEPTensor
@end

static NSUInteger ANEPDtypeBytes(NSString *d) {
    if ([d isEqualToString:@"fp16"]) return 2;
    if ([d isEqualToString:@"fp32"]) return 4;
    return 0;
}

static ANEPTensor *ANEPTensorFromMatch(NSTextCheckingResult *m, NSString *s, NSUInteger dtypeG, NSUInteger shapeG, NSString *name) {
    ANEPTensor *t = [ANEPTensor new];
    t.name = name;
    t.dtype = [s substringWithRange:[m rangeAtIndex:dtypeG]];
    t.shapeText = [s substringWithRange:[m rangeAtIndex:shapeG]];
    NSUInteger prod = 1;
    for (NSString *p in [t.shapeText componentsSeparatedByString:@","]) {
        NSString *tp = [p stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
        if (tp.length == 0) {
            continue;
        }
        prod *= (NSUInteger)MAX(1, tp.integerValue);
    }
    t.elements = prod;
    t.elemBytes = ANEPDtypeBytes(t.dtype);
    return t;
}

// Index just past the ')' matching the '(' at `open`, or NSNotFound.
static NSUInteger ANEPMatchParen(NSString *s, NSUInteger open) {
    int depth = 0;
    for (NSUInteger i = open; i < s.length; i++) {
        unichar c = [s characterAtIndex:i];
        if (c == '(') depth++;
        else if (c == ')') {
            depth--;
            if (depth == 0) return i + 1;
        }
    }
    return NSNotFound;
}

static NSArray<NSString *> *ANEPSplitTop(NSString *s) {
    NSMutableArray *items = [NSMutableArray array];
    int depth = 0;
    NSUInteger start = 0;
    for (NSUInteger i = 0; i < s.length; i++) {
        unichar c = [s characterAtIndex:i];
        if (c == '<' || c == '[' || c == '(' || c == '{') depth++;
        else if (c == '>' || c == ']' || c == ')' || c == '}') depth--;
        else if (c == ',' && depth == 0) {
            [items addObject:[s substringWithRange:NSMakeRange(start, i - start)]];
            start = i + 1;
        }
    }
    [items addObject:[s substringFromIndex:start]];
    NSMutableArray *out = [NSMutableArray array];
    for (NSString *it in items) {
        NSString *t = [it stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
        if (t.length) [out addObject:t];
    }
    return out;
}

// Fills ins/outs from the `func main` signature. Returns NO with *why set when it cannot.
static BOOL ANEPParseMainSignature(NSString *mil, NSMutableArray<ANEPTensor *> *ins,
                                   NSMutableArray<ANEPTensor *> *outs, NSString **why) {
    NSRange fr = [mil rangeOfString:@"func main"];
    if (fr.location == NSNotFound) {
        *why = @"no 'func main' in MIL";
        return NO;
    }
    NSRange po = [mil rangeOfString:@"(" options:0 range:NSMakeRange(NSMaxRange(fr), mil.length - NSMaxRange(fr))];
    if (po.location == NSNotFound) {
        *why = @"no parameter list after 'func main'";
        return NO;
    }
    NSUInteger pend = ANEPMatchParen(mil, po.location);
    if (pend == NSNotFound) {
        *why = @"unbalanced parameter list";
        return NO;
    }
    NSString *params = [mil substringWithRange:NSMakeRange(po.location + 1, pend - po.location - 2)];
    NSRegularExpression *tre = [NSRegularExpression regularExpressionWithPattern:
                                @"^tensor<\\s*(\\w+)\\s*,\\s*\\[([^\\]]*)\\]\\s*>\\s+(\\w+)$" options:0 error:nil];
    for (NSString *item in ANEPSplitTop(params)) {
        NSTextCheckingResult *m = [tre firstMatchInString:item options:0 range:NSMakeRange(0, item.length)];
        if (!m) {
            *why = [NSString stringWithFormat:@"input is not a plain tensor: '%@'", ANEPTrunc(item, 80)];
            return NO;
        }
        [ins addObject:ANEPTensorFromMatch(m, item, 1, 2, [item substringWithRange:[m rangeAtIndex:3]])];
    }
    NSRange arrow = [mil rangeOfString:@"->" options:0 range:NSMakeRange(pend, mil.length - pend)];
    if (arrow.location == NSNotFound) {
        *why = @"no '->' output list";
        return NO;
    }
    NSRange oo = [mil rangeOfString:@"(" options:0 range:NSMakeRange(NSMaxRange(arrow), mil.length - NSMaxRange(arrow))];
    if (oo.location == NSNotFound) {
        *why = @"no output list";
        return NO;
    }
    NSUInteger oend = ANEPMatchParen(mil, oo.location);
    if (oend == NSNotFound) {
        *why = @"unbalanced output list";
        return NO;
    }
    NSString *outList = [mil substringWithRange:NSMakeRange(oo.location + 1, oend - oo.location - 2)];
    for (NSString *name in ANEPSplitTop(outList)) {
        NSString *pat = [NSString stringWithFormat:@"tensor<\\s*(\\w+)\\s*,\\s*\\[([^\\]]*)\\]\\s*>\\s+%@\\s*=",
                         [NSRegularExpression escapedPatternForString:name]];
        NSRegularExpression *re = [NSRegularExpression regularExpressionWithPattern:pat options:0 error:nil];
        NSTextCheckingResult *m = [re firstMatchInString:mil options:0 range:NSMakeRange(0, mil.length)];
        if (!m) {
            *why = [NSString stringWithFormat:@"could not find the tensor definition of output '%@'", name];
            return NO;
        }
        [outs addObject:ANEPTensorFromMatch(m, mil, 1, 2, name)];
    }
    return YES;
}

#pragma mark - environment (stage a)

static void ANEPStageEnvironment(ANEPLog *L) {
    NSProcessInfo *pi = [NSProcessInfo processInfo];
    NSString *platform =
#if TARGET_OS_SIMULATOR
    @"ios-simulator";
#elif TARGET_OS_IPHONE
    @"ios-device";
#else
    @"macos";
#endif
    [L log:@"env: platform=%@ os=\"%@\" hw.machine=%@ hw.model=%@ arch=%@ physmem=%llu MB",
     platform, pi.operatingSystemVersionString, ANEPSysctl("hw.machine"), ANEPSysctl("hw.model"),
#if defined(__arm64__)
     @"arm64",
#elif defined(__x86_64__)
     @"x86_64",
#else
     @"other",
#endif
     pi.physicalMemory / (1024ULL * 1024ULL)];
    const char *simName = getenv("SIMULATOR_DEVICE_NAME");
    if (simName) {
        [L log:@"env: SIMULATOR_DEVICE_NAME=%s", simName];
    }
    [L log:@"env: NSTemporaryDirectory=%@", NSTemporaryDirectory()];

    NSArray<NSString *> *names = @[@"_ANEClient", @"_ANEInMemoryModelDescriptor", @"_ANEInMemoryModel",
                                   @"_ANERequest", @"_ANEIOSurfaceObject", @"_ANEDeviceInfo"];
    [L log:@"env: before dlopen: _ANEClient %@", NSClassFromString(@"_ANEClient") ? @"present" : @"absent"];
    void *h = dlopen(kANEFrameworkPath, RTLD_LAZY);
    if (h) {
        [L log:@"env: dlopen(AppleNeuralEngine) succeeded"];
    } else {
        const char *de = dlerror();
        [L log:@"env: dlopen(AppleNeuralEngine) failed: %s (expected to be unnecessary on iOS)", de ? de : "(no dlerror)"];
    }
    for (NSString *n in names) {
        [L log:@"env: class %@ : %@", n, NSClassFromString(n) ? @"present" : @"MISSING"];
    }

    Class di = NSClassFromString(@"_ANEDeviceInfo");
    if (!di) {
        [L log:@"env: _ANEDeviceInfo missing, skipping device info"];
    } else {
        @try {
            [L log:@"env: _ANEDeviceInfo class methods %@", ANEPMethodNames(di, YES, nil, 2500)];
        } @catch (NSException *ex) {
            [L log:@"env: EXCEPTION listing _ANEDeviceInfo methods: %@ %@", ex.name, ex.reason];
        }
        // Selector names are best guesses (not present in the reference code); each is guarded.
        NSArray<NSString *> *sels = @[@"hasANE", @"numANEs", @"numANECores", @"aneArchitectureType",
                                      @"aneSubType", @"aneSubTypeVariant", @"aneSubTypeProductVariant",
                                      @"aneBoardType", @"isVirtualMachine", @"isInternalBuild",
                                      @"productName", @"buildVersion", @"bootArgs"];
        for (NSString *s in sels) {
            @try {
                NSString *r = ANEPDescribeCall((id)di, s, nil);
                [L log:@"env: +[_ANEDeviceInfo %@] = %@", s, r ?: @"(selector not present)"];
            } @catch (NSException *ex) {
                [L log:@"env: EXCEPTION calling +[_ANEDeviceInfo %@]: %@ %@", s, ex.name, ex.reason];
            }
        }
    }

    Class cl = NSClassFromString(@"_ANEClient");
    if (!cl) {
        [L log:@"env: _ANEClient missing, skipping sharedConnection"];
    } else {
        @try {
            NSString *r = ANEPDescribeCall((id)cl, @"sharedConnection", nil);
            [L log:@"env: +[_ANEClient sharedConnection] = %@", r ?: @"(selector not present)"];
        } @catch (NSException *ex) {
            [L log:@"env: EXCEPTION in +[_ANEClient sharedConnection]: %@ %@", ex.name, ex.reason];
        }
    }
}

#pragma mark - per-model context

@interface ANEPCtx : NSObject
@property (nonatomic, copy) NSString *path;
@property (nonatomic, copy) NSString *label;
@property (nonatomic, copy) NSString *mil;
@property (nonatomic, strong) NSData *milData;
@property (nonatomic, strong) NSData *weightData;
@property (nonatomic, strong) id model;
@property (nonatomic, copy) NSString *tmpDir;
@property (nonatomic) BOOL compiled;
@property (nonatomic) BOOL loaded;
@property (nonatomic) unsigned int qos;
@property (nonatomic, copy) NSString *evalResult;   // @"0", @"1" or @"skipped"
@end
@implementation ANEPCtx
@end

static void ANEPRemoveTmp(ANEPCtx *c, ANEPLog *L) {
    if (c.tmpDir.length) {
        NSError *e = nil;
        BOOL ok = [[NSFileManager defaultManager] removeItemAtPath:c.tmpDir error:&e];
        [L log:@"[%@] cleanup: removed temp dir %@ : %@", c.label, c.tmpDir, ok ? @"yes" : ANEPErr(e)];
        c.tmpDir = nil;
    }
}

#pragma mark - stage b: read, compile, load

static BOOL ANEPReadModel(ANEPCtx *c, ANEPLog *L) {
    NSFileManager *fm = [NSFileManager defaultManager];
    BOOL isDir = NO;
    if (![fm fileExistsAtPath:c.path isDirectory:&isDir] || !isDir) {
        [L log:@"[%@] model path is not a directory: %@", c.label, c.path];
        return NO;
    }
    NSArray *top = [fm contentsOfDirectoryAtPath:c.path error:nil];
    [L log:@"[%@] mlmodelc entries: %@", c.label, [top componentsJoinedByString:@" "]];
    NSString *milPath = [c.path stringByAppendingPathComponent:@"model.mil"];
    NSString *wPath = [c.path stringByAppendingPathComponent:@"weights/weight.bin"];
    c.milData = [NSData dataWithContentsOfFile:milPath];
    if (!c.milData) {
        [L log:@"[%@] cannot read %@ (is this an ML Program compiled model?)", c.label, milPath];
        return NO;
    }
    c.mil = [[NSString alloc] initWithData:c.milData encoding:NSUTF8StringEncoding];
    c.weightData = [NSData dataWithContentsOfFile:wPath options:NSDataReadingMappedIfSafe error:nil];
    [L log:@"[%@] model.mil bytes=%lu (utf8 %@)  weights/weight.bin bytes=%@", c.label,
     (unsigned long)c.milData.length, c.mil ? @"ok" : @"INVALID",
     c.weightData ? [NSString stringWithFormat:@"%lu", (unsigned long)c.weightData.length] : @"missing"];
    if (c.mil) {
        [L log:@"[%@] MIL head (400 chars):", c.label];
        [L log:@"%@", [c.mil substringToIndex:MIN((NSUInteger)400, c.mil.length)]];
    }
    return YES;
}

static void ANEPCompile(ANEPCtx *c, ANEPLog *L) {
    Class Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class IMM = NSClassFromString(@"_ANEInMemoryModel");
    SEL selDesc = NSSelectorFromString(@"modelWithMILText:weights:optionsPlist:");
    SEL selModel = NSSelectorFromString(@"inMemoryModelWithDescriptor:");
    SEL selCompile = NSSelectorFromString(@"compileWithQoS:options:error:");
    SEL selHex = NSSelectorFromString(@"hexStringIdentifier");
    if (!Desc || !IMM) {
        [L log:@"[%@] compile: classes missing (descriptor=%d model=%d)", c.label, Desc != nil, IMM != nil];
        return;
    }
    if (![(id)Desc respondsToSelector:selDesc] || ![(id)IMM respondsToSelector:selModel]) {
        [L log:@"[%@] compile: selector missing (modelWithMILText:weights:optionsPlist: %d, inMemoryModelWithDescriptor: %d)",
         c.label, [(id)Desc respondsToSelector:selDesc], [(id)IMM respondsToSelector:selModel]];
        return;
    }
    // Weight dictionary offset: the macOS reference uses 64, the iPhone reference 0 (with its own header
    // included in the blob). Try 64 first and retry with 0 when compile fails.
    NSArray<NSNumber *> *offsets = c.weightData ? @[@64, @0] : @[@0];
    for (NSNumber *off in offsets) {
        NSDictionary *wdict = c.weightData
            ? @{@"@model_path/weights/weight.bin": @{@"offset": off, @"data": c.weightData}}
            : @{};
        id desc = ((id (*)(Class, SEL, id, id, id))objc_msgSend)(Desc, selDesc, c.milData, wdict, nil);
        [L log:@"[%@] descriptor (weights offset %@): %@", c.label, off, desc ? @"created" : @"nil"];
        if (!desc) {
            continue;
        }
        id model = ((id (*)(Class, SEL, id))objc_msgSend)(IMM, selModel, desc);
        [L log:@"[%@] model object: %@", c.label, model ? NSStringFromClass([model class]) : @"nil"];
        if (!model) {
            continue;
        }
        NSString *hex = nil;
        if ([model respondsToSelector:selHex]) {
            id h = ((id (*)(id, SEL))objc_msgSend)(model, selHex);
            if ([h isKindOfClass:[NSString class]]) {
                hex = h;
            }
        }
        if (hex.length == 0) {
            hex = [[NSUUID UUID] UUIDString];
            [L log:@"[%@] hexStringIdentifier unavailable, using random dir name %@", c.label, hex];
        }
        NSFileManager *fm = [NSFileManager defaultManager];
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hex];
        NSError *werr = nil;
        BOOL dirOK = [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
                   withIntermediateDirectories:YES attributes:nil error:&werr];
        BOOL milOK = dirOK && [c.milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        BOOL wOK = YES;
        if (dirOK && c.weightData) {
            wOK = [c.weightData writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];
        }
        [L log:@"[%@] temp dir %@ : dir=%d mil=%d weights=%d %@", c.label, td, dirOK, milOK, wOK,
         dirOK ? @"" : ANEPErr(werr)];
        c.tmpDir = td;
        c.model = model;

        if (![model respondsToSelector:selCompile]) {
            [L log:@"[%@] compile: model does not respond to compileWithQoS:options:error:", c.label];
            return;
        }
        NSError *err = nil;
        uint64_t t0 = mach_absolute_time();
        BOOL ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(model, selCompile, c.qos, @{}, &err);
        double dt = ANEPSeconds(mach_absolute_time() - t0);
        if (ok) {
            [L log:@"[%@] compile: YES in %.3f s (qos %u, weights offset %@)%@", c.label, dt, c.qos, off,
             err ? [@" with NSError: " stringByAppendingString:ANEPErr(err)] : @""];
            c.compiled = YES;
            return;
        }
        [L log:@"[%@] compile: NO after %.3f s (qos %u, weights offset %@): %@", c.label, dt, c.qos, off, ANEPErr(err)];
        ANEPRemoveTmp(c, L);
        c.model = nil;
    }
}

static void ANEPLoad(ANEPCtx *c, ANEPLog *L) {
    SEL selLoad = NSSelectorFromString(@"loadWithQoS:options:error:");
    if (![c.model respondsToSelector:selLoad]) {
        [L log:@"[%@] load: model does not respond to loadWithQoS:options:error:", c.label];
        return;
    }
    NSError *err = nil;
    uint64_t t0 = mach_absolute_time();
    BOOL ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(c.model, selLoad, c.qos, @{}, &err);
    double dt = ANEPSeconds(mach_absolute_time() - t0);
    if (ok) {
        [L log:@"[%@] load: YES in %.3f s%@", c.label, dt, err ? [@" with NSError: " stringByAppendingString:ANEPErr(err)] : @""];
        c.loaded = YES;
    } else {
        [L log:@"[%@] load: NO after %.3f s: %@", c.label, dt, ANEPErr(err)];
    }
}

static void ANEPIntrospect(ANEPCtx *c, ANEPLog *L) {
    id m = c.model;
    [L log:@"[%@] model class %@ ; related instance methods %@", c.label, NSStringFromClass([m class]),
     ANEPMethodNames([m class], NO, @"ymbol|ttribute|rogram|state|nput|output|queue|perf|handle", 1500)];
    NSArray<NSString *> *plain = @[@"state", @"modelAttributes", @"hexStringIdentifier", @"programHandle",
                                   @"intermediateBufferHandle", @"queueDepth", @"perfStatsMask", @"isMILModel",
                                   @"modelURL", @"string"];
    for (NSString *s in plain) {
        @try {
            NSString *r = ANEPDescribeCall(m, s, nil);
            [L log:@"[%@] -[model %@] = %@", c.label, s, r ?: @"(selector not present)"];
        } @catch (NSException *ex) {
            [L log:@"[%@] EXCEPTION in -[model %@]: %@ %@", c.label, s, ex.name, ex.reason];
        }
    }
    for (NSString *s in @[@"inputSymbolIndicesForProcedureIndex:", @"outputSymbolIndicesForProcedureIndex:"]) {
        @try {
            NSString *r = ANEPDescribeCall(m, s, @0);
            [L log:@"[%@] -[model %@0] = %@", c.label, s, r ?: @"(selector not present)"];
        } @catch (NSException *ex) {
            [L log:@"[%@] EXCEPTION in -[model %@0]: %@ %@", c.label, s, ex.name, ex.reason];
        }
    }
}

#pragma mark - stage c: evaluate

static void ANEPEvaluate(ANEPCtx *c, ANEPLog *L) {
    c.evalResult = @"skipped";
    if (!c.mil) {
        [L log:@"[%@] eval skipped: MIL text not readable", c.label];
        return;
    }
    NSMutableArray<ANEPTensor *> *ins = [NSMutableArray array];
    NSMutableArray<ANEPTensor *> *outs = [NSMutableArray array];
    NSString *why = nil;
    if (!ANEPParseMainSignature(c.mil, ins, outs, &why)) {
        [L log:@"[%@] eval skipped: could not parse the main signature (%@)", c.label, why];
        return;
    }
    for (ANEPTensor *t in ins) {
        [L log:@"[%@] MIL input  %@ : %@ [%@]", c.label, t.name, t.dtype, t.shapeText];
    }
    for (ANEPTensor *t in outs) {
        [L log:@"[%@] MIL output %@ : %@ [%@]", c.label, t.name, t.dtype, t.shapeText];
    }
    if (ins.count != 1 || outs.count != 1) {
        [L log:@"[%@] eval skipped: model has %lu inputs and %lu outputs; only single-input single-output models are evaluated "
               @"(the input/output order the ANE expects for multi-tensor models is not known yet)",
         c.label, (unsigned long)ins.count, (unsigned long)outs.count];
        return;
    }
    ANEPTensor *tin = ins[0], *tout = outs[0];
    if (tin.elemBytes == 0 || tout.elemBytes == 0) {
        [L log:@"[%@] eval skipped: unsupported dtype (in %@, out %@)", c.label, tin.dtype, tout.dtype];
        return;
    }

    Class AR = NSClassFromString(@"_ANERequest");
    Class AIO = NSClassFromString(@"_ANEIOSurfaceObject");
    SEL selReq = NSSelectorFromString(@"requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:");
    SEL selObj = NSSelectorFromString(@"objectWithIOSurface:");
    SEL selEval = NSSelectorFromString(@"evaluateWithQoS:options:request:error:");
    if (!AR || !AIO || ![(id)AR respondsToSelector:selReq] || ![(id)AIO respondsToSelector:selObj]
        || ![c.model respondsToSelector:selEval]) {
        [L log:@"[%@] eval skipped: request/IOSurface/evaluate selectors unavailable", c.label];
        return;
    }
    c.evalResult = @"0";

    NSUInteger inBytes = tin.elements * tin.elemBytes;
    NSUInteger outBytes = tout.elements * tout.elemBytes;
    IOSurfaceRef sIn = ANEPMakeSurface(inBytes);
    IOSurfaceRef sOut = ANEPMakeSurface(outBytes);
    if (!sIn || !sOut) {
        [L log:@"[%@] eval: IOSurfaceCreate failed (in=%d out=%d)", c.label, sIn != NULL, sOut != NULL];
        if (sIn) CFRelease(sIn);
        if (sOut) CFRelease(sOut);
        return;
    }
    [L log:@"[%@] eval: input surface %lu bytes (payload %lu), output surface %lu bytes (payload %lu)", c.label,
     (unsigned long)MAX(inBytes, kMinSurfaceBytes), (unsigned long)inBytes,
     (unsigned long)MAX(outBytes, kMinSurfaceBytes), (unsigned long)outBytes];

    // Known input pattern, exactly representable in fp16: ((i % 17) - 8) / 8.
    float *expect = (float *)malloc(tin.elements * sizeof(float));
    IOSurfaceLock(sIn, 0, NULL);
    uint8_t *ip = (uint8_t *)IOSurfaceGetBaseAddress(sIn);
    memset(ip, 0, MAX(inBytes, kMinSurfaceBytes));
    for (NSUInteger i = 0; i < tin.elements; i++) {
        float v = (float)((int)(i % 17) - 8) / 8.0f;
        expect[i] = v;
        if (tin.elemBytes == 2) {
            ((uint16_t *)ip)[i] = ANEPFloatToHalf(v);
        } else {
            ((float *)ip)[i] = v;
        }
    }
    IOSurfaceUnlock(sIn, 0, NULL);

    id wIn = ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(AIO, selObj, sIn);
    id wOut = ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(AIO, selObj, sOut);
    id req = nil;
    if (wIn && wOut) {
        req = ((id (*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(AR, selReq, @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);
    }
    if (!req) {
        [L log:@"[%@] eval: could not build _ANEIOSurfaceObject/_ANERequest (in=%d out=%d)", c.label, wIn != nil, wOut != nil];
    } else {
        NSError *err = nil;
        uint64_t t0 = mach_absolute_time();
        BOOL ok = ((BOOL (*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(c.model, selEval, c.qos, @{}, req, &err);
        double first = ANEPSeconds(mach_absolute_time() - t0);
        if (!ok) {
            [L log:@"[%@] eval: first evaluate FAILED after %.3f s: %@", c.label, first, ANEPErr(err)];
        } else {
            [L log:@"[%@] eval: first evaluate OK in %.4f s", c.label, first];
            IOSurfaceLock(sOut, kIOSurfaceLockReadOnly, NULL);
            const uint8_t *op = (const uint8_t *)IOSurfaceGetBaseAddress(sOut);
            NSMutableString *vals = [NSMutableString string];
            NSUInteger nz = 0;
            double maxDiff = 0;
            NSUInteger n = tout.elements;
            for (NSUInteger i = 0; i < n; i++) {
                float v = tout.elemBytes == 2 ? ANEPHalfToFloat(((const uint16_t *)op)[i]) : ((const float *)op)[i];
                if (v != 0.0f) nz++;
                if (i < 8) [vals appendFormat:@"%g ", (double)v];
                if (n == tin.elements) {
                    double d = fabs((double)v - (double)expect[i]);
                    if (isnan(d)) d = INFINITY;
                    if (d > maxDiff) maxDiff = d;
                }
            }
            IOSurfaceUnlock(sOut, kIOSurfaceLockReadOnly, NULL);
            [L log:@"[%@] eval: first 8 output values: %@", c.label, vals];
            [L log:@"[%@] eval: nonzero outputs %lu of %lu%@", c.label, (unsigned long)nz, (unsigned long)n,
             n == tin.elements ? [NSString stringWithFormat:@"; max |out-in| = %g (0 expected for an identity conv)", maxDiff] : @""];
            c.evalResult = @"1";

            const int warm = 3, iters = 50;
            BOOL allOK = YES;
            for (int i = 0; i < warm && allOK; i++) {
                allOK = ((BOOL (*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(c.model, selEval, c.qos, @{}, req, &err);
            }
            uint64_t t1 = mach_absolute_time();
            int done = 0;
            for (int i = 0; i < iters && allOK; i++) {
                allOK = ((BOOL (*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(c.model, selEval, c.qos, @{}, req, &err);
                if (allOK) done++;
            }
            double tot = ANEPSeconds(mach_absolute_time() - t1);
            if (allOK) {
                [L log:@"[%@] eval: mean time over %d evaluations: %.4f ms", c.label, done, tot / MAX(done, 1) * 1000.0];
            } else {
                [L log:@"[%@] eval: repeated evaluate failed after %d ok runs: %@", c.label, done, ANEPErr(err)];
            }
        }
    }
    free(expect);
    CFRelease(sIn);
    CFRelease(sOut);
}

#pragma mark - stage d: cleanup

static void ANEPCleanup(ANEPCtx *c, ANEPLog *L) {
    if (c.loaded && c.model) {
        SEL selUnload = NSSelectorFromString(@"unloadWithQoS:error:");
        if ([c.model respondsToSelector:selUnload]) {
            NSError *err = nil;
            BOOL ok = ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(c.model, selUnload, c.qos, &err);
            [L log:@"[%@] cleanup: unload %@%@", c.label, ok ? @"YES" : @"NO", ok ? @"" : [@" : " stringByAppendingString:ANEPErr(err)]];
        } else {
            [L log:@"[%@] cleanup: model does not respond to unloadWithQoS:error:", c.label];
        }
        c.loaded = NO;
    }
    ANEPRemoveTmp(c, L);
    c.model = nil;
}

#pragma mark - public

NSString *ANEProbeEnvironment(void (^logLine)(NSString *line)) {
    ANEPLog *L = [[ANEPLog alloc] initWithSink:logLine];
    [L log:@"START environment"];
    @try {
        ANEPStageEnvironment(L);
    } @catch (NSException *ex) {
        [L log:@"EXCEPTION in environment stage: %@ %@", ex.name, ex.reason];
    }
    [L log:@"END environment"];
    return [L text];
}

NSString *ANEProbeModel(NSString *mlmodelcPath, NSString *label, void (^logLine)(NSString *line)) {
    ANEPLog *L = [[ANEPLog alloc] initWithSink:logLine];
    ANEPCtx *c = [ANEPCtx new];
    c.path = mlmodelcPath;
    c.label = label.length ? label : @"model";
    c.qos = 21;   // macOS reference uses 21, the iPhone reference 9; override with ANEPROBE_QOS
    const char *q = getenv("ANEPROBE_QOS");
    if (q && atoi(q) > 0) {
        c.qos = (unsigned int)atoi(q);
    }
    c.evalResult = @"skipped";
    [L log:@"START model %@ path=%@ qos=%u", c.label, c.path, c.qos];

    BOOL haveModel = NO;
    @try {
        haveModel = ANEPReadModel(c, L);
    } @catch (NSException *ex) {
        [L log:@"[%@] EXCEPTION reading model: %@ %@", c.label, ex.name, ex.reason];
    }
    if (haveModel) {
        @try {
            ANEPCompile(c, L);
        } @catch (NSException *ex) {
            [L log:@"[%@] EXCEPTION in compile stage: %@ %@", c.label, ex.name, ex.reason];
        }
    }
    if (c.compiled) {
        @try {
            ANEPLoad(c, L);
        } @catch (NSException *ex) {
            [L log:@"[%@] EXCEPTION in load stage: %@ %@", c.label, ex.name, ex.reason];
        }
    }
    if (c.loaded) {
        @try {
            ANEPIntrospect(c, L);
        } @catch (NSException *ex) {
            [L log:@"[%@] EXCEPTION in introspection: %@ %@", c.label, ex.name, ex.reason];
        }
        @try {
            ANEPEvaluate(c, L);
        } @catch (NSException *ex) {
            [L log:@"[%@] EXCEPTION in evaluate stage: %@ %@", c.label, ex.name, ex.reason];
            c.evalResult = @"0";
        }
    } else if (haveModel) {
        [L log:@"[%@] eval skipped: model not loaded", c.label];
    }
    @try {
        ANEPCleanup(c, L);
    } @catch (NSException *ex) {
        [L log:@"[%@] EXCEPTION in cleanup: %@ %@", c.label, ex.name, ex.reason];
    }
    [L log:@"DONE %@ compile=%d load=%d eval=%@", c.label, c.compiled ? 1 : 0, c.loaded ? 1 : 0, c.evalResult];
    return [L text];
}

NSString *ANEProbeRun(NSString *mlmodelcPath, NSString *label, void (^logLine)(NSString *line)) {
    NSString *a = ANEProbeEnvironment(logLine);
    NSString *b = ANEProbeModel(mlmodelcPath, label, logLine);
    return [a stringByAppendingString:b];
}

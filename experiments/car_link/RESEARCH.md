# Car link from an iPhone: what can reach a panda (or a bridge to the car)

Research date: 2026-11-16. Question: how can an iOS app on an iPhone (USB-C, A19, current iOS) exchange about 100 small messages per second each way, ideally under 10-20 ms latency, with a panda or an intermediate device on the car?

This document establishes facts; it does not pick a design.

Evidence labels used throughout:

- **[P]** confirmed by a primary source (Apple documentation, Apple Accessory Design Guidelines, comma source code). Apple DTS engineer replies on developer.apple.com/forums are Apple staff speaking officially but informally; they are marked **[P-DTS]**.
- **[3P]** reported by third parties (forum posts, blogs, vendor pages).
- **[INF]** my inference; not directly stated by any source.
- **[NONE]** I looked and found no evidence either way.

Sources fetched during this research (local copies, not committed): Apple Accessory Design Guidelines "Release R32 Update 1" (PDF footer date 2026-10-30), commaai/panda master (commit e957e2a, 2026-11-16, cloned with history), commaai/openpilot master (commit 21f7747, 2026-11-16), Apple developer documentation pages, Apple developer forum threads.

---

## 1. Summary table

| # | Option | Works on iPhone? | Apple approval / special entitlement? | Extra hardware | Expected latency | Evidence strength |
|---|--------|------------------|----------------------------------------|----------------|------------------|-------------------|
| 1a | Direct USB to the panda (vendor-specific class 0xFF) via DriverKit / USBDriverKit | **No.** DriverKit is iPadOS (M-series) and macOS only | n/a (not available) | none | n/a | Strong [P-DTS], repeated 2022-2025 |
| 1b | Direct USB via IOKit / IOUSBHost from a dev-signed app | **No supported path.** DTS: "I/O Kit on non-Mac platforms is only rated for talking to a DriverKit driver". No evidence anyone does it on iPhone | Would need restricted entitlements that a normal provisioning profile does not grant [P-DTS, indirect] | none | n/a | Strong that it is unsupported [P-DTS]; **no evidence either way** that a private-API hack works on a non-jailbroken iPhone |
| 1c | ExternalAccessory / MFi (iAP2) directly to the panda | **No.** Panda does not speak iAP2 and has no Apple authentication chip | MFi licence, Apple auth coprocessor, iAP2 firmware | MFi-capable bridge (panda cannot be one) | Unknown | Strong [P-DTS] that EA needs an MFi accessory |
| 2a | USB Ethernet (CDC-NCM) bridge, plain sockets (UDP/TCP) | **Yes.** Apple-supported, no MFi, public Network.framework / BSD sockets | None found. App needs Local Network permission prompt in some configurations | Bridge that is a USB NCM gadget and talks to panda or CAN (ESP32-S3 per DTS, or a Linux SBC in gadget mode) | Not published. Third-party report of a 30 ms batching burst on one adapter; otherwise unmeasured | Support: strong [P-DTS + P]. Latency: **none** |
| 2b | USB MIDI (CoreMIDI), arbitrary bytes via SysEx | **Likely yes.** Apple guidelines confirm class-compliant USB MIDI works with Core MIDI. Using it for non-music data: [NONE] from Apple | None found | Bridge that is a USB-MIDI gadget | Unmeasured on iPhone | Support for MIDI: [P]. Use for arbitrary data: [3P]/[INF] |
| 2c | USB audio | Supported for audio only | None | Bridge presenting as audio device | n/a | [P]; **not a data path** [INF] |
| 2d | USB HID | iOS handles HID itself (keyboards, game controllers). No generic HID API for apps on iPhone found | CoreHID virtual-device API needs an Apple-approved entitlement and is documented for macOS only | Bridge as HID gadget | n/a | Weak. See section 2d |
| 2e | USB mass storage | Exposed as files, not as a message channel | None | Bridge as storage gadget | Not suitable [INF] | [INF] |
| 2f | USB serial (CDC-ACM, FTDI, etc.) | **No.** "iOS doesn't allow you to write your own code to add that support"; serial exposed only via MFi cables | MFi (e.g. Redpark cable is MFi iAP2 to serial) | Redpark-type MFi cable | Unknown | Strong [P-DTS] |
| 3a | Bluetooth LE (CoreBluetooth) to a bridge | **Yes.** No Apple approval needed | None for the app. Bridge is any BLE peripheral | BLE-capable bridge (e.g. ESP32 or nRF) | Interval chosen by iPhone, not app. Apple's accessory rules put the minimum at 15 ms (11.25 ms for HID). One third-party iOS 18.6.2 report saw a fixed 30 ms interval and batching | Guidelines [P]; real latency [3P], weak |
| 3b | Wi-Fi (infrastructure or soft-AP) to a bridge | **Yes.** NEHotspotConfiguration (iOS 11+) joins a named network; sockets are public | Hotspot Configuration capability (standard, not MFi) [INF: not verified in this research] | Wi-Fi bridge (ESP32 or Linux SBC) | Unmeasured here | Mechanism [P]; latency [NONE] |
| 3c | Wi-Fi Aware (iOS 26+) | Yes on iPhone 12 and later, iOS 26+ | `com.apple.developer.wifi-aware` entitlement | Wi-Fi Aware certified peer | Apple says "low-latency", no numbers | API [P]; hardware availability for hobby bridges [NONE] |
| 3d | Panda's own wireless | **No current panda has it.** White panda had ESP8266 Wi-Fi; removed Aug 2020 | n/a | n/a | n/a | Removal [P] (git history); reason [NONE] |
| 4 | Existing projects | No project found that connects a panda to an iPhone or iPad. Android projects use USB OTG directly | n/a | n/a | n/a | See section 4 |

Panda side (Option 5): the panda is a USB device with a vendor-specific interface (bulk EP1 IN, EP3 OUT, control requests for safety mode and heartbeat). It does not authenticate the host. It drops to SILENT if no heartbeat arrives for 2 s (ignition off) or 5 s (ignition on). Current openpilot's own C++ `pandad` is SPI-only; the USB path lives in the panda Python library.

---

## 2. Three most important findings

1. **The "iPhone USB-C can talk to the panda" assumption is false for any app-level path I could find.** The panda is a vendor-specific (class 0xFF) USB device. On iPhone there is no public or DTS-sanctioned way to talk to such a device. Apple DTS, September 2025: "DriverKit is not supported on iOS." and "General USB communication is not possible, even through the MFi program." [P-DTS: https://developer.apple.com/forums/thread/801333]
2. **The only non-MFi wired data path Apple explicitly endorses is USB Ethernet (NCM).** Apple DTS (Kevin Elliott), May and July 2026, said an ESP32-S3 acting as an NCM device "basically turns it into an Ethernet over USB accessory", says no other hardware is needed ("you don't actually need ANY other hardware"), and that Apple's Ethernet support is "NOT 'accidental'" though Apple cannot make formal commitments [P-DTS: https://developer.apple.com/forums/thread/772812]. The Apple Accessory Design Guidelines (R32 Update 1, 2026-10-30) have a chapter "Ethernet over USB" requiring NCM v1.0 [P]. That means a bridge is needed between the iPhone and the panda; the iPhone cannot host the panda directly.
3. **Nothing about the panda's firmware depends on the host being a comma device, but the host must keep a heartbeat and set the safety mode over a vendor control request.** I found no host authentication in the `0xdc` (set safety mode) or `0xf3` (heartbeat) handlers [P: panda `board/main_comms.h`]. A bridge that speaks the panda USB protocol (or SPI) can substitute for the host. Whether openpilot's safety design assumes a comma host is a separate question; the firmware enforces limits itself ([P] panda README).

---

## 3. Option sections

### Option 1. Direct USB to a vendor-specific device

#### 1a. DriverKit / USBDriverKit

Facts:

- DriverKit on iPadOS exists from iPadOS 16, with USBDriverKit, PCIDriverKit, AudioDriverKit, and "requires an iPad with an M-series chip" [P: https://developer.apple.com/documentation/driverkit/creating-drivers-for-ipados.md, page copyright 2026]. The same page documents entitlements `com.apple.developer.driverkit.communicates-with-drivers` and `...allow-third-party-userclients` for iPadOS.
- Apple DTS, Sept 2024: "DriverKit is currently only supported on macOS and iPadOS, not iOS (or iPhones)." [P-DTS: https://developer.apple.com/forums/thread/763677]
- Apple DTS, Sept 2025: "Correct. DriverKit is not supported on iOS." and the website's "iOS 16.0+" metadata is explained as API overlap: "our website doesn't do a very good job handling edge cases like an iPadOS-only API." [P-DTS: https://developer.apple.com/forums/thread/801333]
- Apple DTS, Feb 2025 and Oct 2025: "DriverKit isn't available on iOS" / "we have been informed DriverKit is only available on iPadOS, not iOS. That's correct." [P-DTS: https://developer.apple.com/forums/thread/773694, https://developer.apple.com/forums/thread/802640]
- Oct 2024, Apple DTS: "No, we have not announced anything new in this area." (asked about non-MFi iPhone USB-C communication). [P-DTS: https://developer.apple.com/forums/thread/766367]
- A developer reported that a DriverKit target embedded in an app on an iPhone 15 Pro Max (iOS 17) was never seen or installed by the system; another saw provisioning errors ("This provisioning profile cannot be installed on this device", Jan 2024) [3P: https://developer.apple.com/forums/thread/737447]. A company (Nov 2023) said Apple told them their iPad USBDriverKit driver on iPhone 15 "is not supported" [3P, same thread].
- The most recent evidence I found for iPhone is September-October 2025 and later DTS replies in 2026 (thread 772812, May and Jul 2026) which still direct iPhone developers to NCM / MFi rather than DriverKit [P-DTS]. I found **no** announcement of DriverKit on iPhone for iOS 26 or iOS 27. [NONE for any later change; check WWDC material if needed.]
- Even on iPadOS, DTS describes DriverKit as having "no way for an app to run in the background" and being "far more complicated" than ExternalAccessory [P-DTS: thread 766367]. This is irrelevant to iPhone but shows the iPad route is not a shortcut.

#### 1b. IOKit / IOUSBHost used privately from a development-signed app

Facts:

- Apple DTS, June 2024: "I/O Kit on non-Mac platforms is only rated for talking to a DriverKit driver. You can't use it to talk to generic hardware." and "On iOS (as opposed to iPadOS) there are only two ways to work with accessories over USB: If the accessory is supported by iOS itself, you can use high-level APIs... If not, your only option is MFi." [P-DTS: https://developer.apple.com/forums/thread/756657]
- Apple DTS, Oct 2025: "I/O Kit is an implementation detail on iOS", and raw USB in userspace is "Not on iOS." (possible via I/O Kit on macOS, DriverKit on iPadOS) [P-DTS: https://developer.apple.com/forums/thread/802640]
- Apple DTS, mid 2025: "IOKit is really only available on iPadOS, NOT iOS." [P-DTS: https://developer.apple.com/forums/thread/793915]
- Entitlements: Apple DTS (Oct 2024) says DriverKit "is dependent on entitlements and those entitlements must be authorised by a provisioning profile." [P-DTS: https://developer.apple.com/forums/thread/743878]. A forum summary reports that claiming a restricted entitlement not authorised by the profile prevents the app from launching (AMFI) [3P: https://developer.apple.com/forums/thread/701980 and neighbours]. This is consistent with Apple's documented model but I did not find a primary Apple page stating it for iPhone sideloading specifically.
- Sandbox: on iPad, IOKit property reads on `IOUSBHostDevice` were denied by the sandbox in at least one report (Jan 2024) [3P: https://developer.apple.com/forums/thread/743878]. That is iPadOS, not iPhone.
- On macOS, raw USB capture through IOUSBHost needs `com.apple.vm.device-access`, which Apple DTS says was designed for hypervisors [3P summary of https://developer.apple.com/forums/thread/701980]. Not applicable to iOS but shows the pattern: raw device access is gated by Apple-granted entitlements.

What I did **not** find: any report of a non-jailbroken iPhone with a development-signed build successfully opening a USB device through IOKit/IOUSBHost, with or without private entitlements. [NONE] I also did not find any Apple statement on whether private IOKit calls are blocked at sandbox or kernel-MACF level on iPhone specifically. My inference [INF] is that, given Apple's statements and the AMFI behaviour reported for restricted entitlements, this is very unlikely to work without a jailbreak or exploit, but the question cannot be settled from documents. The smallest hands-on test is listed in section 6.

#### 1c. ExternalAccessory framework and MFi

- ExternalAccessory talks only to accessories "built under the aegis of the MFi programme"; the accessory must implement iAP2; EA "doesn't expose a USB-like API" [P-DTS: https://developer.apple.com/forums/thread/802640, https://developer.apple.com/forums/thread/793915]. Apple's guidelines describe the EA protocol as a read/write bytestream between app and accessory [P: Accessory Design Guidelines R32U1, chapter 38 "External Accessory Protocol"].
- Apple DTS: "General USB communication is not possible, even through the MFi program." [P-DTS: thread 801333]. So MFi would not make the panda itself usable; it would only make a purpose-built MFi bridge possible.
- Redpark is the cited example of a ready-made MFi serial path: its cable contains an Apple authentication chip and iAP2 firmware, converting iAP2 traffic to serial [3P: https://developer.apple.com/forums/thread/795202, a Redpark employee and Apple engineer]. DTS (Jan 2025) pointed hobbyists at "a third-party hobbyist solution" and named Redpark [P-DTS: https://developer.apple.com/forums/thread/772812]. Apple's MFi page says: "We recommend using a third-party hobbyist solution to connect iOS devices to serial devices, and to write apps that communicate with these serial devices." [P: https://mfi.apple.com/en/who-should-join]. A forum summary says the Redpark SDK is limited to apps deployed through Apple Business Manager / School Manager [3P: https://developer.apple.com/forums/thread/737447]; I did not verify this against Redpark's own documentation.
- Apple's guidance to commercial accessory makers: join MFi and learn what is involved [P-DTS: thread 766367]. Individuals making accessories for personal use are listed under "Who does not need to join" [P: mfi.apple.com/en/who-should-join], but that does not provide a way to build an iAP2 accessory without the licence and authentication chip [INF].
- Developer experience warning: DTS recommends UIKit and the older app lifecycle for ExternalAccessory and says SwiftUI/scene lifecycle "may need to require iOS 26" [P-DTS: thread 793915]. A beta-era third-party report says EASession initialisation returns nil on iOS 27 beta with an MFi USB accessory while working on iOS 26 [3P, unconfirmed: https://developer.apple.com/forums/tags/externalaccessory].

### Option 2. USB classes iOS handles natively

#### 2a. USB Ethernet (CDC-NCM; also ECM)

Facts:

- Apple's Accessory Design Guidelines: "Accessories may support Ethernet over USB using the Network Control Model (NCM) protocol... shall comply with the Network Control Model Devices Specification - Revision 1.0" [P: R32U1, chapter 37, p.256, dated 2026-10-30].
- Apple support lists "USB-to-Ethernet adapters" among devices usable on the iPhone USB-C port (iPhone 15 and later) [P: https://support.apple.com/en-us/105099].
- Apple DTS (Kevin Elliott), Jan 2025: Ethernet supported "LONG (iOS 10.2?)"; USB-C Ethernet adapters work [P-DTS: https://developer.apple.com/forums/thread/772812].
- Apple DTS, May 2026: for an ESP32-S3 as an NCM device, "you don't actually need ANY other hardware"; it "basically turns it into an Ethernet over USB accessory." [P-DTS: same thread]
- Apple DTS, Jul 2026 (custom Cortex-R8 / DWC3 NCM device, iOS 18.x): no feasibility issues raised; iOS "doesn't require a DHCP-assigned address and is perfectly happy to operate with a network that consists entirely of self-assigned local addresses"; use Network.framework; link-local addressing may trigger the Local Network privacy authorization (TN3179), which a minimal DHCP plus DNS plus gateway setup might avoid; Apple cannot make formal commitments about the future but Ethernet support is "NOT 'accidental'", disabling it "would be far more disruptive than you might think"; "Network adaptors are generally not MFi licensed because there isn't anything to license"; no special App Store review issues expected for standard socket use [P-DTS: same thread].
- Apple DTS, Oct 2025: a non-standard chip (AX88772C) is not supported by the built-in driver; recommended path is to change the vendor firmware to work with "iOS's built-in USB Ethernet driver" or use an off-the-shelf dongle [P-DTS: https://developer.apple.com/forums/thread/802640]. So a bridge must present a standard class, not a vendor-specific Ethernet protocol.
- Throughput/latency: Apple DTS "doesn't have enough experience at that level" to answer throughput (Jan 2025) [P-DTS: thread 772812]. A third-party developer sending 1 ms UDP packets reported the Apple USB Ethernet driver sometimes accumulating about 30 ms of data and releasing it in a burst [3P, from a search summary of https://developer.apple.com/forums/thread/668146; I did not open the thread to confirm]. An informal musician report says wired Ethernet from an iPad was lower latency than USB audio, "<5 ms" [3P, anecdote]. No controlled iPhone measurement found. [NONE]

What a bridge looks like in principle [INF]:

- A device that is a USB NCM (or ECM) gadget toward the iPhone and holds an IP stack. Candidates: ESP32-S3 (named by Apple DTS), or a Linux SBC with USB gadget mode (not named by Apple).
- On the other side it needs to reach the car: either (a) act as USB host to the panda and relay the panda's bulk/control protocol (ESP32-S3 has a USB host; Linux SBC with libusb runs the existing `panda` Python library), or (b) bypass the panda with its own CAN transceiver, which would lose the panda firmware's safety enforcement unless the bridge reimplements it.
- The app sees ordinary UDP/TCP sockets. Open points: Local Network permission, whether iOS treats the link-local interface as primary when Wi-Fi/cellular are also up (Apple points to NWParameters interface controls), and the app lifetime in the background (see section 6).

#### 2b. USB MIDI (CoreMIDI)

Facts:

- Apple Guidelines: "Compatible USB and Bluetooth Low Energy MIDI accessories can interface directly with apps using the Core MIDI framework." USB accessories must implement a MIDI Streaming IN endpoint and comply with USB MIDI 1.0 or 2.0 [P: R32U1, chapter 43, dated 2026-10-30].
- A community post (Dec 2023) suggests MIDI SysEx as a carrier for arbitrary data over USB-C iPhones: "It should work just fine for any type of devices, not just musical instruments", while calling it "not ideal" [3P: https://developer.apple.com/forums/thread/721063]. A Feb 2025 forum question asking whether using CoreMIDI with a non-MFi class-compliant accessory could violate App Store policy had no answer shown [NONE: same thread]. Since the target is a development-signed build, App Store policy is not the constraint [INF].
- Same thread, Apple DTS (Dec 2022): for serial, "iOS doesn't allow you to write your own code to add that support", suggests a BLE bridge [P-DTS].
- Latency: I found no measurement of CoreMIDI over USB on iPhone. A 2005 Mac post reports 2-3 ms round trips on cheap interfaces and a 2011 Apple list post argues USB-MIDI has "absolutely no control over timing or latency" [3P, old and off-platform]. [NONE for iPhone].
- Payload: MIDI 1.0 over USB uses 4-byte event packets; a SysEx carries 7-bit data per byte [INF from the MIDI spec, not verified in this research], so an encoding layer is needed.

Bridge in principle [INF]: a microcontroller with a USB-MIDI class gadget (ESP32-S3, RP2040 etc.) mapping SysEx to the panda protocol or CAN frames.

#### 2c. USB audio

- Apple lists audio accessories as supported on the iPhone 15+ USB-C port [P: https://support.apple.com/en-us/105099], and Apple's guidelines reference USB Audio Class 2.0/4.0 codecs for USB-C [P: R32U1 line ~4175].
- Using an audio path as a modem for CAN data would be a hack with isochronous framing and buffering; I found no evidence of anyone doing it, and it is unlikely to meet 10-20 ms with reliability [INF].

#### 2d. USB HID

- DTS (June 2024): for accessories iOS supports natively, "you can work with a keyboard using both UIKit and Game Controller framework." [P-DTS: thread 756657]. That is not raw access to arbitrary HID reports.
- CoreHID (HIDDeviceManager, HIDDeviceClient, virtual devices): documentation metadata lists "macOS: 15.0.0 -" only [P: https://developer.apple.com/documentation/corehid/hiddevicemanager.md]. The virtual-device API needs `com.apple.developer.hid.virtual.device`, requested via Apple's DriverKit entitlement form [3P summary of DTS replies, https://developer.apple.com/forums/thread/843327 and others]. One developer reported a request pending 2.5 months [3P: https://developer.apple.com/forums/thread/845599].
- A forum search result says a DTS reply pointed to CoreHID as the "modern API" for an HID access question; I could not tie that to iPhone [3P, unverified].
- Net: **no evidence of a general raw-HID channel for iPhone apps.** [NONE for any app-visible HID report API on iPhone]. The "Game Controller" framework could in principle carry button/axis state from a gamepad-class gadget, but that is a one-way, low-bandwidth, not-designed-for-this path [INF].

#### 2e. Mass storage

- Apple lists external storage as supported [P: support.apple.com/en-us/105099]. Apps access such storage through system file APIs, not as a message channel. A bridge using a shared file as a mailbox would add latency and caching problems [INF]; I found no sources about it.

#### 2f. Serial (CDC-ACM, FTDI, CP210x)

- Apple DTS, Jan 2025: serial is "Not in the way you're looking for"; "the list of supported classes doesn't include serial." (macOS does) [P-DTS: https://developer.apple.com/forums/thread/772812]
- Apple DTS, Dec 2022: "for iOS there doesn't seem to be support for serial communication" is "basically correct"; iOS "doesn't allow you to write your own code to add that support" [P-DTS: https://developer.apple.com/forums/thread/721063]
- Apple DTS, Jan 2025 also said serial adapter solutions were a better option than DriverKit even on iPadOS [P-DTS: thread 772812]. On iPhone that means MFi hobbyist cables such as Redpark.
- So CDC-ACM-based dongles (including typical USB-CAN adapters based on serial) will not give an iPhone app access. [P-DTS].

### Option 3. Wireless

#### 3a. Bluetooth LE via CoreBluetooth

Facts from Apple's guidelines (R32U1, 2026-10-30) [P]:

- General connection-parameter guidance for accessories (section 57.6): Peripheral latency <= 30 intervals; supervision timeout 6-18 s; **Interval Min >= 15 ms and a multiple of 15 ms**; Interval Max at least 15 ms above Interval Min, or both 15 ms; Interval Max x (latency+1) <= 6 s; supervision timeout > Interval Max x (latency+1) x 3; requests that do not meet the rules "may be rejected". "If Bluetooth Low Energy HID is one of the connected services of an accessory, a connection interval down to 11.25 ms may be accepted by some devices."
- Spatial-accessory chapter (19.4.7.3) lets the iPhone as Central pick 15 ms or 30 ms intervals; the app cannot dictate them [P].
- CoreBluetooth does not expose the negotiated interval to apps (2016 mailing-list report) [3P, old].
- L2CAP channels (`CBL2CAPChannel`) exist from iOS 11 and give a stream interface [P: https://developer.apple.com/documentation/corebluetooth/cbl2capchannel.md].
- Background: with `UIBackgroundModes` = `bluetooth-central`, the app can keep communicating with a connected peripheral in the background and is woken for delegate callbacks, with about 10 s of runtime per wake [P, archived page dated 2013: https://developer.apple.com/library/archive/documentation/NetworkingInternetWeb/Conceptual/CoreBluetooth_concepts/CoreBluetoothBackgroundProcessingForIOSApps/PerformingTasksWhileYourAppIsInTheBackground.html]. Current behaviour should be rechecked.

Third-party reports of real behaviour [3P]:

- iPhone 15 Pro, iOS 18.6.2, Raspberry Pi 5 peripheral, 20-byte writes without response every 25 ms: btmon showed a **fixed 30 ms connection interval**; iOS appeared to batch writes/notifications and latency was noticeably higher than Android. No Apple reply. Single report, Nov 2025: https://developer.apple.com/forums/thread/806188
- Throughput reports: iOS 18.2, 400-600 kbit/s unstable versus 1000-1200 kbit/s on an iPhone 13 (https://developer.apple.com/forums/thread/773555); an Apple engineer advice is to use write-without-response and multiple writes per interval (summary of forum posts, not fetched individually).
- With a 15 ms interval you cannot get better than roughly one connection event per 15 ms per direction; a 100 Hz (10 ms) cadence is therefore faster than the connection events [INF from the interval floor], so messages would be batched two or more per event. Whether that meets the target depends on the latency budget; no measurement found.
- BLE-MIDI is mentioned in Apple's guidelines; one vendor-adjacent test reports 12.3 ms median round trip on an iPad Pro M2 [3P, weak, from a search summary].

Bridge in principle [INF]: any BLE peripheral (ESP32, nRF52) that connects to the panda over USB host or SPI, or to CAN directly.

#### 3b. Wi-Fi (infrastructure or soft-AP)

- `NEHotspotConfiguration` (iOS 11+) joins a Wi-Fi network by SSID or prefix from an app [P: https://developer.apple.com/documentation/networkextension/nehotspotconfiguration.md]. Capability/entitlement needs and join prompts were not verified in this research. [INF: standard capability, not MFi.]
- iPhone OBD-II apps commonly use ELM327-style Wi-Fi or BLE adapters; Wi-Fi adapters require joining the adapter's own network, which developers describe as troublesome, and BLE is the simpler choice [3P: e.g. https://www.carscanner.info/category/settings/]. Classic Bluetooth adapters do not work without MFi [3P: same].
- No Wi-Fi latency measurements gathered. [NONE]
- Wi-Fi Aware (iOS 26+, iPhone 12 and later) gives peer-to-peer Wi-Fi with "high-bandwidth and low-latency data transfers", requires a Wi-Fi Aware certified accessory and the `com.apple.developer.wifi-aware` entitlement [P: https://developer.apple.com/documentation/wifiaware.md]. Availability of hobbyist Wi-Fi Aware hardware: [NONE found].
- A wired Ethernet link is the NCM path in 2a. Apple's guidelines and DTS replies cover it as a "network accessory", which avoids Wi-Fi association problems [INF].

#### 3c. Panda's own wireless support

- The original white panda (STM32F413 plus ESP8266): "On the computer side, it has both USB and Wi-Fi... connected over high speed SPI, so the panda is actually capable of dumping the full contents of the busses over Wi-Fi" [P: panda README at commit 8b41ed3^, from `git show`]. The Python library had a `WifiHandle` that tunnelled the USB request protocol over a TCP socket to 192.168.0.10:1337 [P: python/__init__.py at that commit].
- Removal: commit "Deprecate ESP (#592)", 2020-08-26, by Adeeb Shihadeh: removed `boardesp`, `esptool`, wifi tests, ESP certs, "no more wifi". "Cleanup Wi-Fi leftovers (#804)", 2021-12-06 [P: panda git history].
- **Stated reason: none found.** The commit messages give no reason, and I did not find a comma statement of safety motivation for the removal. [NONE]. The old README did say "Currently, setting safety modes is only supported over USB" [P: README at 8b41ed3^], which indicates the wireless path was at least restricted in what it could do, but that is a documented limitation, not a stated rationale.
- Current boards in panda master are `red`, `tres`, `cuatro` (`board/boards/`). `red` has USB only (`has_spi = false`); `tres` and `cuatro` have SPI (comma 3X / comma four internal) [P: board/boards/*.h]. None has wireless. [P]
- A third-party GitHub README claims "The hardware interface called Panda is compatible with iOS (at least over WiFi)"; that refers to the old white panda era and cannot be true of current hardware [3P: https://raw.githubusercontent.com/fotidim/openpilot-coreml/master/README.md; contradicted by the commits above].
- Historical iOS: a ~2016 spec listing said the panda supported iOS Wi-Fi ELM327 apps [3P, old, from a search summary; not opened].

### Option 4. Existing projects

- **flowpilot** (Android/PC fork of openpilot): needs "a white / grey panda with giraffe or a black/red panda with car harness" and "an OTG cable is required if connecting panda to phone" [P: https://raw.githubusercontent.com/flowdriveai/flowpilot/master/README.md]. A repo diff describes the no-root Android path as using Termux's `termux-usb` to obtain a USB file descriptor from Android and passing it to `boardd`, which wraps it in libusb [3P: https://gh.r-mj.com/flowdriveai/flowpilot/pull/92.diff, from a search summary; I did not open it]. Android lets apps receive a USB permission/file descriptor for any device; iOS has no equivalent [INF].
- **fotidim/openpilot-coreml**: converts the `supercombo` model to Core ML. Its README states the glue code "needs to be ran or ported to iOS" and makes the Wi-Fi claim above. Looks abandoned (0 stars/forks in the search summary) [3P].
- **iOS OBD-II apps** (e.g. Car Scanner, Engine Link, LeafSpy, OBD2 Expert) use ELM327-compatible BLE or Wi-Fi adapters [3P: App Store listings via search]. No wired USB-C CAN adapter app for iPhone found [NONE].
- **Any project connecting a panda to an iPhone or iPad:** none found. [NONE]
- **iPad:** DriverKit on M-series iPads would make a dext for the panda technically possible per Apple's docs [P: creating-drivers-for-ipados], but it needs Apple-approved DriverKit entitlements, and I found no example for the panda. Not applicable to iPhone.

### Option 5. The panda side

Source: commaai/panda master, commit e957e2a (2026-11-16); commaai/openpilot master, commit 21f7747 (2026-11-16). All [P] unless stated.

USB interface:

- Device descriptor class/subclass/protocol 0xFF/0xFF/0xFF, VID 0x3801 (comma) with PID 0xDDEE (release) or 0xDDCC (bootstub/debug); the Python library also accepts VID 0xBBAA (`board/config.h`, `board/drivers/usb.h`, `python/__init__.py`).
- One interface, alt 0 with three **bulk** endpoints: EP1 IN (CAN receive), EP2 OUT (bootstub flashing), EP3 OUT (CAN send), 64-byte max packet. Alt setting 1 changes EP1 to an **interrupt** endpoint with polling interval 5 frames. The descriptors say USB 2.1 with 64-byte bulk packets, which indicates full speed [INF].
- BOS descriptor exposes WebUSB and WinUSB (`WEBUSB_VENDOR_CODE 0x30`) [`board/drivers/usb.h`]. This helps browsers on desktop; it has no relevance to iOS, which has no WebUSB in Safari [INF: not verified here].
- Control requests are vendor-type, device-recipient. Key ones (from `python/__init__.py` and `board/main_comms.h`): `0xdc` set safety mode (wValue mode, wIndex param); `0xf3` heartbeat (wValue 1 = engaged); `0xf8` disable heartbeat checks (refused in car safety modes); `0xd2` health struct; `0xde` CAN speed; `0xf1` clear queue; `0xc1` hardware type; `0xd0` serial; `0xdd` packet-layout hashes. Full list in the library source.
- CAN packet framing: 6-byte header (DLC<<4 | bus<<1 | FD; 32-bit little-endian `address<<3 | extended<<2`; XOR checksum) then data; receive-side "returned" and "rejected" flags in the header [`python/__init__.py`, `selfdrive/pandad/panda.cc`]. Struct layouts are checked against hashes of `can.h` and `health.h` at connect time.
- SPI: the same logical endpoints and control requests run over SPI on `tres` and `cuatro` [`openpilot/selfdrive/pandad/panda_comms.h`, `board/boards/*.h`].

Heartbeat and safety behaviour (`board/main.c`):

- Panda powers up in SAFETY_SILENT; no CAN is transmitted until the host sets a safety mode with request `0xdc` (README, firmware).
- A 1 Hz counter increments; if it reaches **5 s with ignition on or 2 s with ignition off** (`HEARTBEAT_IGNITION_CNT_ON 5`, `_OFF 2`) and checks are not disabled, the firmware sets `heartbeat_lost`, clears the engaged state, and switches to SAFETY_SILENT (with a siren countdown if it was engaged). A heartbeat (`0xf3`) resets the counter and re-enables checks.
- openpilot's `pandad` runs a 100 Hz loop, receiving CAN each pass, sends the heartbeat at 10 Hz with the engaged flag, and a separate thread sends CAN as needed; it sets safety mode from CarParams once on-road (`pandad.cc`, `panda_safety.cc`). Those numbers give a reference load: about 100 bulk reads per second plus 10 control writes per second plus event-driven CAN sends [P].
- `0xf8` (disable heartbeat) is blocked in car safety modes, so a bridge cannot rely on it for car control [P: `board/main_comms.h`].

Host assumptions:

- In the `0xdc`, `0xf3` and CAN bulk handlers I saw **no authentication of the host**. The firmware identifies itself by serial/secret (`0xd0`) but does not verify who is talking to it [P, from reading the handlers; I did not audit the whole firmware].
- Safety enforcement (torque and rate limits, `controls_allowed` from the car's own cruise/engagement messages) is in the panda firmware via opendbc safety code [P: panda README: "provides and enforces the" openpilot safety requirements; opendbc `opendbc/safety/` contains the per-car logic]. A bridge that forwards the unchanged panda protocol keeps these protections; one that replaces the panda with a bare CAN transceiver does not [INF].
- **Current openpilot's C++ `pandad` is SPI-only**: `Panda::list()` returns `PandaSpiHandle::list()` and `panda_comms.h` contains only the SPI handle [P]. The USB path lives in the Python `panda` library (`python/usb.py`, `usb1`/libusb) [P]. So reusing openpilot's `pandad` unchanged through a USB bridge is not possible; a port would reimplement `pandad`'s loop using the Python library's protocol or the `PandaSpiHandle` protocol [INF]. A `SocketPanda` class that wraps Linux SocketCAN exists in the library (since 2024-11, #2073), with `set_safety_mode` as a no-op [P: `python/socketpanda.py`], which shows the library tolerates non-panda CAN backends but without the panda's safety firmware.

---

## 4. Direct answers to the numbered investigation points

- 1. Direct USB from an iOS app to a vendor-specific device on iPhone: no public API [P-DTS]. DriverKit: iPadOS 16+ on M-series only [P]; not iPhone [P-DTS, latest 2025, consistent through 2026 replies]. ExternalAccessory/MFi: iAP2 only, requires Apple-licensed accessory [P-DTS]. IOKit/IOUSBHost privately: Apple says unsupported; no evidence of anyone doing it on iPhone; nothing found either way on whether it can work in a dev-signed build.
- 2. Native classes: Ethernet NCM works with sockets [P]; USB MIDI works with CoreMIDI [P], arbitrary payload via SysEx suggested by third parties; audio works; HID is handled by iOS (no generic raw HID for apps found); mass storage is file-level; serial is not exposed to apps [P-DTS].
- 3. Wireless: BLE (interval floor 15 ms in Apple's rules, iPhone decides), Wi-Fi/NEHotspotConfiguration, Wi-Fi Aware (iOS 26+). Old panda Wi-Fi removed in Aug 2020; no stated reason found.
- 4. Existing work: Android projects use USB OTG directly; no iOS/iPad panda project found.
- 5. Panda protocol and heartbeat: see Option 5 in section 3.

---

## 5. Things that depend on the iOS version or date

- DriverKit-on-iPhone status is stated by Apple staff on 2024-09, 2025-02, 2025-09, 2025-10; the NCM statements are from Jan, May, and Jul 2026 [P-DTS]. A change at a later iOS release would not show up here. [NONE for iOS 26/27 announcements].
- Wi-Fi Aware: iOS 26+ [P].
- ExternalAccessory with SwiftUI / scene lifecycle: DTS says iOS 26 may be required [P-DTS, 2025].
- A forum thread reports an EASession nil on iOS 27 beta with an MFi USB accessory [3P, unconfirmed beta report].
- Apple Accessory Design Guidelines are R32 Update 1 dated 2026-10-30. The BLE parameter text in the older R17 release said 11.25 ms may be accepted for HID [3P blog quoting R17], matching the current guidance.
- One result in my searches (a 2023 article about Apple limiting speeds on non-MFi USB-C cables) was a leaker rumour with no detail; I treat it as unreliable and did not use it.

---

## 6. Open questions that only hands-on testing can settle

1. Can any app on a normal development-signed iPhone build see or open a vendor-class USB device? (IOKit/IOUSBHost private use). Documents say no; there is no evidence for or against a private-API workaround.
2. What are the actual round-trip latency distribution (median, p99, max) and jitter of a 100 Hz request/response exchange over: (a) USB NCM to an app UDP socket on iPhone 17, (b) USB MIDI SysEx via CoreMIDI, (c) BLE (write without response plus notify, or L2CAP), (d) Wi-Fi soft-AP? Nothing authoritative is published.
3. Does the iPhone batch or delay small UDP packets over the NCM interface (the reported 30 ms bursts)? Does it depend on the gadget's NCM aggregation settings?
4. Which BLE connection interval does an iPhone 17 actually choose for a custom peripheral (15 ms, 30 ms)? Can an app influence it? Does iOS batch notifications as reported?
5. Does a USB Ethernet link with self-assigned addressing, no gateway, and Wi-Fi/cellular active route the app's UDP to the gadget reliably, and does it trigger the Local Network prompt each time?
6. What happens when the app is backgrounded, screen locks, or iOS suspends it? The heartbeat deadline is 2 s with ignition off and 5 s with ignition on, so any suspension longer than that drops the panda to SILENT. A driving app in the foreground with the idle timer disabled may avoid this, but background behaviour for each transport was not verified. [NONE for USB sockets; archived Apple page for BLE].
7. Can a bridge (ESP32-S3 or Linux SBC) sustain 100 Hz bulk reads plus 10 Hz heartbeats against a red panda's USB interface, and what is its added latency?
8. USB power: how a bridge powered from the iPhone port and the panda powered from the car interact (ground loops, back-powering). Not researched.
9. Will future iOS releases keep third-party NCM gadgets working? Apple declined to commit [P-DTS, Jul 2026].

### Smallest tests

- **Test A (settles the direct-USB assumption, minutes):** plug the panda into the iPhone via USB-C. Build a minimal app that (i) tries `ExternalAccessory` `EAAccessoryManager.shared().connectedAccessories` and (ii) tries to link and call IOKit `IOServiceGetMatchingServices` for `IOUSBHostDevice`. Expected from Apple's statements: empty accessory list and sandbox denial or no usable services. This gives a hands-on yes/no for 1b. Keep the panda unpowered from the car; this test sends no CAN.
- **Test B (settles the Ethernet path, a day):** flash an ESP32-S3 as a USB NCM gadget with a UDP echo, write an iPhone app (Network.framework, foreground, idle timer disabled) that sends 100 small datagrams per second for 10 minutes with timestamps, and log round-trip median, p99, max, and gaps. Repeat with the app backgrounded. Then add the panda behind the ESP32 (USB host) or substitute a Linux SBC with the panda Python library.
- **Test C (cheap comparison):** the same 100 Hz echo over BLE (peripheral requests 15 ms; record the interval with a sniffer or the peripheral's log) and over CoreMIDI SysEx using the same ESP32-S3 firmware family.
- Neither B nor C need the car.

---

## 7. Source list

Apple documentation and guidelines:
- Creating drivers for iPadOS: https://developer.apple.com/documentation/driverkit/creating-drivers-for-ipados.md
- Accessory Design Guidelines for Apple Devices, Release R32 Update 1 (footer 2026-10-30): https://developer.apple.com/accessories/Accessory-Design-Guidelines.pdf (chapters 37 Ethernet over USB, 38 External Accessory Protocol, 43 MIDI, 57.6 BLE connection parameters, 19.4.7.3)
- iPhone USB-C connector support page: https://support.apple.com/en-us/105099
- Wi-Fi Aware: https://developer.apple.com/documentation/wifiaware.md
- NEHotspotConfiguration: https://developer.apple.com/documentation/networkextension/nehotspotconfiguration.md
- CBL2CAPChannel: https://developer.apple.com/documentation/corebluetooth/cbl2capchannel.md
- CoreHID HIDDeviceManager (macOS 15 only in metadata): https://developer.apple.com/documentation/corehid/hiddevicemanager.md
- Core Bluetooth background processing (archived 2013): https://developer.apple.com/library/archive/documentation/NetworkingInternetWeb/Conceptual/CoreBluetooth_concepts/CoreBluetoothBackgroundProcessingForIOSApps/PerformingTasksWhileYourAppIsInTheBackground.html
- MFi "Who should join": https://mfi.apple.com/en/who-should-join

Apple DTS forum replies (primary but informal):
- DriverKit not on iOS (Sep 2025): https://developer.apple.com/forums/thread/801333
- DriverKit iPadOS only (Sep 2024): https://developer.apple.com/forums/thread/763677
- IOKit on non-Mac (Jun 2024): https://developer.apple.com/forums/thread/756657
- IOKit on iOS (Feb 2025): https://developer.apple.com/forums/thread/773694
- Custom USB network device (Oct 2025): https://developer.apple.com/forums/thread/802640
- ESP32 to iPhone 16, NCM guidance (Jan 2025, May 2026, Jul 2026): https://developer.apple.com/forums/thread/772812
- Serial communication (Dec 2022 onward): https://developer.apple.com/forums/thread/721063
- USB-C without MFi (Oct 2024): https://developer.apple.com/forums/thread/766367
- External Accessory (Jul 2025): https://developer.apple.com/forums/thread/793915
- Entitlements and signing (Oct 2024): https://developer.apple.com/forums/thread/743878
- Redpark and serial (iPad thread): https://developer.apple.com/forums/thread/795202

Third-party reports:
- DriverKit on USB-C iPhones: https://developer.apple.com/forums/thread/737447
- BLE batching on iOS 18.6.2: https://developer.apple.com/forums/thread/806188
- BLE throughput on iOS 18.2: https://developer.apple.com/forums/thread/773555
- USB Ethernet batching report: https://developer.apple.com/forums/thread/668146
- CoreHID entitlement request: https://developer.apple.com/forums/thread/845599
- OBD apps use BLE/Wi-Fi: https://www.carscanner.info/category/settings/
- flowpilot README: https://raw.githubusercontent.com/flowdriveai/flowpilot/master/README.md
- flowpilot Termux path diff: https://gh.r-mj.com/flowdriveai/flowpilot/pull/92.diff
- openpilot-coreml: https://raw.githubusercontent.com/fotidim/openpilot-coreml/master/README.md

comma sources (cloned and read locally):
- panda: https://github.com/commaai/panda (python/__init__.py, python/socketpanda.py, board/main.c, board/main_comms.h, board/drivers/usb.h, board/boards/*.h; history: commits 8b41ed3 "Deprecate ESP (#592)" and 493866d "Cleanup Wi-Fi leftovers (#804)")
- openpilot: https://github.com/commaai/openpilot (selfdrive/pandad/pandad.cc, panda.cc, panda_comms.h, panda_safety.cc)
- opendbc: https://github.com/commaai/opendbc (opendbc/safety/)

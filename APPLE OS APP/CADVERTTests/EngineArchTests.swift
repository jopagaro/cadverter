#if os(macOS)
import XCTest
@testable import CADVERT

/// The bundled engine must match the architecture of the app slice running it. Shipping an
/// arm64 engine inside an Intel build (or vice versa) is invisible until a user launches it,
/// so assert the relationship here.
final class EngineArchTests: XCTestCase {

    func testCurrentArchMatchesTheRunningProcess() {
        // What the process actually is, independent of the compile-time check in EngineLocator.
        var sysctlArch = [CChar](repeating: 0, count: 64)
        var size = sysctlArch.count
        let known = sysctlbyname("hw.machine", &sysctlArch, &size, nil, 0) == 0
        XCTAssertTrue(["arm64", "x86_64"].contains(EngineLocator.currentArch),
                      "unexpected arch \(EngineLocator.currentArch)")
        if known {
            let machine = String(cString: sysctlArch)
            // Under Rosetta hw.machine reports x86_64, which is exactly the slice we are.
            XCTAssertEqual(EngineLocator.currentArch, machine,
                           "EngineLocator.currentArch must match the running slice")
        }
    }

    /// If this build bundles an engine, its interpreter must contain our architecture.
    func testBundledEngineMatchesThisSlice() throws {
        guard let python = EngineLocator.bundledPython() else {
            throw XCTSkip("no engine bundled in the test host")
        }
        let lipo = Process()
        lipo.executableURL = URL(fileURLWithPath: "/usr/bin/lipo")
        lipo.arguments = ["-archs", python.path]
        let pipe = Pipe()
        lipo.standardOutput = pipe
        lipo.standardError = pipe
        try lipo.run()
        lipo.waitUntilExit()
        let archs = String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        XCTAssertTrue(archs.split(separator: " ").map(String.init).contains(EngineLocator.currentArch),
                      "bundled engine is [\(archs)] but this slice is \(EngineLocator.currentArch)")
    }

    /// The mismatch flag is only meaningful when no usable engine was found.
    func testMismatchFlagIsConsistent() {
        if EngineLocator.bundledPython() != nil {
            XCTAssertFalse(EngineLocator.bundleHasOtherArchEngineOnly(),
                           "a usable engine was found, so nothing should report a mismatch")
        }
    }

    /// A custom path still wins, and the bundled engine is only a fallback.
    func testCandidateOrderPrefersCustomPath() {
        let custom = "/tmp/some/python3"
        let list = EngineLocator.candidates(customPath: custom)
        XCTAssertEqual(list.first?.path, custom)
        XCTAssertTrue(list.contains { $0.path.contains(".cadvert/engine") },
                      "~/.cadvert/engine should remain a fallback")
    }
}
#endif

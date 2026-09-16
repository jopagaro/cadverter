#if os(macOS)
import XCTest
@testable import CADVERT

/// Integration test for the Mac "local engine" path. Uses the repo's own .venv
/// (…/CADVERT/.venv/bin/python) when present; skipped otherwise.
final class LocalEngineTests: XCTestCase {
    private func devPython() -> URL? {
        if let env = ProcessInfo.processInfo.environment["CADVERT_TEST_PYTHON"], !env.isEmpty {
            return URL(fileURLWithPath: env)
        }
        // <repo>/APPLE OS APP/CADVERTTests/LocalEngineTests.swift → <repo>/.venv/bin/python
        let here = URL(fileURLWithPath: #filePath)
        let repo = here.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let py = repo.appendingPathComponent(".venv/bin/python")
        return FileManager.default.isExecutableFile(atPath: py.path) ? py : nil
    }

    func testFreePortIsUsable() {
        let p = LocalEngine.freePort()
        XCTAssertGreaterThan(p, 1024)
        XCTAssertLessThan(p, 65536)
    }

    func testLocatorRejectsMissingPython() async {
        let r = await EngineLocator.validate(URL(fileURLWithPath: "/nonexistent/python3"))
        XCTAssertFalse(r.ok)
    }

    func testEngineStartsAndServesConfig() async throws {
        guard let py = devPython() else { throw XCTSkip("no dev python with cadvert installed") }
        let v = await EngineLocator.validate(py)
        guard v.ok else { throw XCTSkip("python at \(py.path) has no cadvert: \(v.detail)") }

        let engine = LocalEngine()
        defer { engine.stop() }
        let base = try await engine.start(python: py, openAIKey: nil, readyTimeout: 90)
        XCTAssertEqual(base.host, "127.0.0.1")

        let client = CadvertClient(baseURL: base)
        let cfg = try await client.fetchConfig()
        XCTAssertEqual(cfg.localOnly, true, "the bundled engine is local-only")

        // Full pipeline on the sample part, exactly what the UI does on drop.
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let sample = repo.appendingPathComponent("samples/test_block_with_holes.step")
        if let data = try? Data(contentsOf: sample) {
            let r = try await client.convert(fileData: data, filename: "test_block_with_holes.step")
            XCTAssertEqual(r.format.uppercased(), "STEP")
            XCTAssertEqual(r.summary.faces, 8)
            XCTAssertTrue(r.hsd.contains("GLOBAL PROPERTIES"))
            XCTAssertFalse(r.images.isEmpty, "renderer should produce views")
            await client.deleteSession(r.sessionId)
        }
        engine.stop()
        XCTAssertFalse(engine.isRunning)
    }
}
#endif

#if os(macOS)
/// End-to-end: Apple's on-device model answering through the real engine's `/tool` endpoint.
/// Skipped unless macOS 26 + Apple Intelligence is available and the dev venv exists.
final class AppleIntelligenceIntegrationTests: XCTestCase {
    func testOnDeviceModelAnswersWithTools() async throws {
        guard AppleIntelligence.isSupportedOS else { throw XCTSkip("no FoundationModels on this OS") }
        let status = AppleIntelligence.status
        guard status.available else { throw XCTSkip("Apple Intelligence unavailable: \(status.detail)") }

        let here = URL(fileURLWithPath: #filePath)
        let repo = here.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let py = repo.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.isExecutableFile(atPath: py.path) else { throw XCTSkip("no dev python") }

        let engine = LocalEngine()
        defer { engine.stop() }
        let base = try await engine.start(python: py, openAIKey: nil, readyTimeout: 90)
        let client = CadvertClient(baseURL: base)
        let data = try Data(contentsOf: repo.appendingPathComponent("samples/test_block_with_holes.step"))
        let r = try await client.convert(fileData: data, filename: "test_block_with_holes.step")
        XCTAssertNotNil(r.tier0, "server should return tier0 for on-device use")
        let part = PartSession(r)

        // Direct tool call first (what the model's tools do under the hood).
        let toolJSON = try await client.executeTool(sessionId: part.id, name: "get_feature", arguments: ["feature_id": "hole_1"])
        XCTAssertTrue(toolJSON.contains("hole") || toolJSON.contains("error"), "tool endpoint answered: \(toolJSON.prefix(120))")

        let ai = AppleIntelligence()
        var text = ""
        var toolCalls: [String] = []
        for try await ev in ai.stream(question: "What are the hole diameters in this part? Use a tool if you need to.", part: part, client: client) {
            switch ev {
            case .content(let c): text += c
            case .toolCall(let t): toolCalls.append(t)
            case .error(let e): XCTFail(e)
            case .done: break
            }
        }
        print("ON-DEVICE ANSWER (tools: \(toolCalls)):\n\(text)\n---")
        XCTAssertFalse(text.isEmpty, "on-device model produced no text")
        await client.deleteSession(part.id)
    }
}
#endif

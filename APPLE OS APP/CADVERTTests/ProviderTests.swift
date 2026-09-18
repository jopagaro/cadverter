import XCTest
@testable import CADVERT

final class ProviderTests: XCTestCase {
    func testChatRequestHeadersPerProvider() throws {
        let client = CadvertClient(baseURL: URL(string: "http://127.0.0.1:9999")!)
        let history = [["role": "user", "content": "hi"]]

        let a = try client.chatRequest(sessionId: "s1", history: history, provider: .anthropic, apiKey: "sk-ant-x", model: "claude-opus-5")
        XCTAssertEqual(a.value(forHTTPHeaderField: "X-Provider"), "anthropic")
        XCTAssertEqual(a.value(forHTTPHeaderField: "X-Model"), "claude-opus-5")
        XCTAssertEqual(a.value(forHTTPHeaderField: "X-Anthropic-Key"), "sk-ant-x")
        XCTAssertNil(a.value(forHTTPHeaderField: "X-OpenAI-Key"))

        let o = try client.chatRequest(sessionId: "s1", history: history, provider: .openai, apiKey: "sk-o", model: "gpt-4o")
        XCTAssertEqual(o.value(forHTTPHeaderField: "X-Provider"), "openai")
        XCTAssertEqual(o.value(forHTTPHeaderField: "X-OpenAI-Key"), "sk-o")
        XCTAssertNil(o.value(forHTTPHeaderField: "X-Anthropic-Key"))

        let noKey = try client.chatRequest(sessionId: "s1", history: history, provider: .openai, apiKey: nil, model: "gpt-4o")
        XCTAssertNil(noKey.value(forHTTPHeaderField: "X-OpenAI-Key"))
        XCTAssertEqual(noKey.url?.path, "/chat/s1")
    }

    func testModelListsAndDefaults() {
        XCTAssertEqual(ChatModel.defaultModel(for: .anthropic), "claude-opus-5")
        XCTAssertEqual(ChatModel.defaultModel(for: .openai), "gpt-5.6-terra")
        XCTAssertTrue(ChatModel.models(for: .anthropic).contains { $0.id == "claude-sonnet-5" })
        XCTAssertTrue(ChatModel.models(for: .openai).contains { $0.id == "gpt-6-astra" })
        // Retired generations must not linger in the picker — a stale list is the whole
        // reason the app asks the provider for a live one.
        XCTAssertFalse(ChatModel.models(for: .openai).contains { $0.id.hasPrefix("gpt-4") })
        XCTAssertTrue(ChatModel.models(for: .anthropic).contains { $0.id == "claude-fable-5-1" })
        XCTAssertTrue(AIProvider.selectable.contains(.openai))
        XCTAssertTrue(AIProvider.selectable.contains(.anthropic))
        // Apple is offered only on hardware that can actually run the on-device model.
        XCTAssertEqual(AIProvider.selectable.contains(.apple), AppleIntelligence.isDeviceCapable)
    }

    func testSettingsKeysPerProvider() {
        let suite = UserDefaults(suiteName: "CADVERTTests.\(UUID().uuidString)")!
        let s = AppSettings(defaults: suite)
        s.provider = .anthropic
        XCTAssertEqual(s.currentModel, "claude-opus-5")
        s.setModel("claude-sonnet-5", for: .anthropic)
        XCTAssertEqual(s.currentModel, "claude-sonnet-5")
        XCTAssertEqual(s.model(for: .openai), "gpt-5.6-terra")
        s.provider = .openai
        XCTAssertEqual(s.currentModel, "gpt-5.6-terra")
        // Apple needs no key
        if AppleIntelligence.isSupportedOS {
            s.provider = .apple
            XCTAssertNil(s.currentKey)
            XCTAssertFalse(s.provider.needsKey)
        }
    }

    func testServerConfigWithProvidersDecodes() throws {
        let json = """
        {"local_only": true, "max_file_mb": 500,
         "providers": {"openai": {"available": true, "key_present": false, "models": ["gpt-4o"], "default_model": "gpt-4o-mini"},
                       "anthropic": {"available": true, "key_present": true, "models": ["claude-opus-5"], "default_model": "claude-opus-5"}},
         "tools": ["get_feature", "measure_distance", "get_component", "compute_mass"]}
        """
        let cfg = try JSONDecoder().decode(ServerConfig.self, from: Data(json.utf8))
        XCTAssertEqual(cfg.localOnly, true)
        XCTAssertEqual(cfg.info(for: .anthropic)?.keyPresent, true)
        XCTAssertEqual(cfg.info(for: .openai)?.defaultModel, "gpt-4o-mini")
        XCTAssertEqual(cfg.tools?.count, 4)

        // An engine that predates these fields must still decode.
        let old = try JSONDecoder().decode(ServerConfig.self, from: Data("{}".utf8))
        XCTAssertNil(old.providers)
    }

    func testAppleStatusIsCoherent() {
        let st = AppleIntelligence.status
        if !AppleIntelligence.isSupportedOS { XCTAssertFalse(st.available) }
        XCTAssertFalse(st.detail.isEmpty)
    }
}

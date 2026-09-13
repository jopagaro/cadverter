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
        XCTAssertEqual(ChatModel.defaultModel(for: .openai), "gpt-4o-mini")
        XCTAssertTrue(ChatModel.models(for: .anthropic).contains { $0.id == "claude-sonnet-5" })
        XCTAssertTrue(ChatModel.models(for: .openai).contains { $0.id == "gpt-4o" })
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
        XCTAssertEqual(s.model(for: .openai), "gpt-4o-mini")
        s.provider = .openai
        XCTAssertEqual(s.currentModel, "gpt-4o-mini")
        // Apple needs no key
        if AppleIntelligence.isSupportedOS {
            s.provider = .apple
            XCTAssertNil(s.currentKey)
            XCTAssertFalse(s.provider.needsKey)
        }
    }

    func testServerConfigWithProvidersDecodes() throws {
        let json = """
        {"disable_auth": true, "stripe_enabled": false, "stripe_byok_enabled": false,
         "providers": {"openai": {"available": true, "server_key": false, "models": ["gpt-4o"], "default_model": "gpt-4o-mini"},
                       "anthropic": {"available": true, "server_key": true, "models": ["claude-opus-5"], "default_model": "claude-opus-5"}},
         "tools": ["get_feature", "measure_distance"]}
        """
        let cfg = try JSONDecoder().decode(ServerConfig.self, from: Data(json.utf8))
        XCTAssertEqual(cfg.info(for: .anthropic)?.serverKey, true)
        XCTAssertEqual(cfg.info(for: .openai)?.defaultModel, "gpt-4o-mini")
        XCTAssertEqual(cfg.tools?.count, 2)
        // Older servers without the block still decode.
        let old = try JSONDecoder().decode(ServerConfig.self, from: Data(#"{"disable_auth": false, "stripe_enabled": false, "stripe_byok_enabled": false}"#.utf8))
        XCTAssertNil(old.providers)
    }

    func testAppleStatusIsCoherent() {
        let st = AppleIntelligence.status
        if !AppleIntelligence.isSupportedOS { XCTAssertFalse(st.available) }
        XCTAssertFalse(st.detail.isEmpty)
        // Hardware that cannot run the model can never report itself ready.
        if !AppleIntelligence.isDeviceCapable {
            XCTAssertFalse(AppleIntelligence.isReadyNow)
            XCTAssertFalse(st.available)
        }
        // Being ready implies being capable.
        if AppleIntelligence.isReadyNow { XCTAssertTrue(AppleIntelligence.isDeviceCapable) }
    }

    /// An Intel Mac runs macOS 26 but can never run Apple Intelligence. The first-run default
    /// must not land on a provider that would fail the moment the user asks a question.
    func testDefaultProviderNeverStartsOnAnUnusableApple() {
        let suite = UserDefaults(suiteName: "CADVERTTests.\(UUID().uuidString)")!
        let s = AppSettings(defaults: suite)
        if s.provider == .apple {
            XCTAssertTrue(AppleIntelligence.isReadyNow,
                          "defaulted to Apple Intelligence on a Mac where it is not ready")
        }
        XCTAssertTrue(AIProvider.selectable.contains(s.provider),
                      "default provider must be one the picker offers")
    }

    /// A stored Apple preference is dropped on hardware that can never honour it.
    func testStoredApplePreferenceDroppedOnIncapableHardware() {
        let suite = UserDefaults(suiteName: "CADVERTTests.\(UUID().uuidString)")!
        suite.set("apple", forKey: "provider")
        let s = AppSettings(defaults: suite)
        if !AppleIntelligence.isDeviceCapable {
            XCTAssertEqual(s.provider, .openai, "incapable hardware must fall back off Apple")
        }
    }
}

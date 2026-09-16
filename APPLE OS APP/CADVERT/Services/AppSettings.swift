import Foundation
import Observation

enum EngineMode: String, CaseIterable, Identifiable {
    /// macOS only: run the Python engine as a child process (bundled, ~/.cadvert/engine, or a custom path).
    case local
    /// Any platform: connect to a hosted `cadvert-server`.
    case remote
    var id: String { rawValue }

    static var available: [EngineMode] {
        #if os(macOS)
        return [.local, .remote]
        #else
        return [.remote]
        #endif
    }
}

/// User preferences (UserDefaults) + API keys (Keychain).
@Observable
final class AppSettings {
    private let defaults: UserDefaults

    var engineMode: EngineMode { didSet { defaults.set(engineMode.rawValue, forKey: "engineMode") } }
    var remoteURLString: String { didSet { defaults.set(remoteURLString, forKey: "remoteURL") } }
    var pythonPath: String { didSet { defaults.set(pythonPath, forKey: "pythonPath") } }
    var theme: ThemePreference { didSet { defaults.set(theme.rawValue, forKey: "theme") } }
    var hasSeenWelcome: Bool { didSet { defaults.set(hasSeenWelcome, forKey: "hasSeenWelcome") } }

    /// Which AI answers questions: Apple's on-device model, OpenAI, or Claude.
    var provider: AIProvider { didSet { defaults.set(provider.rawValue, forKey: "provider") } }
    var openAIModel: String { didSet { defaults.set(openAIModel, forKey: "openAIModel") } }
    var anthropicModel: String { didSet { defaults.set(anthropicModel, forKey: "anthropicModel") } }

    /// The user's own keys (BYOK on the web; the only keys the offline Mac app can use).
    var openAIKey: String { didSet { KeychainStore.save(openAIKey, account: "openai_api_key") } }
    var anthropicKey: String { didSet { KeychainStore.save(anthropicKey, account: "anthropic_api_key") } }

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        #if os(macOS)
        let defaultMode = EngineMode.local
        #else
        let defaultMode = EngineMode.remote
        #endif
        engineMode      = EngineMode(rawValue: defaults.string(forKey: "engineMode") ?? "") ?? defaultMode
        remoteURLString = defaults.string(forKey: "remoteURL") ?? "http://localhost:8000"
        pythonPath      = defaults.string(forKey: "pythonPath") ?? ""
        theme           = ThemePreference(rawValue: defaults.string(forKey: "theme") ?? "") ?? .system
        hasSeenWelcome  = defaults.bool(forKey: "hasSeenWelcome")
        openAIModel     = defaults.string(forKey: "openAIModel") ?? defaults.string(forKey: "model") ?? ChatModel.defaultModel(for: .openai)
        anthropicModel  = defaults.string(forKey: "anthropicModel") ?? ChatModel.defaultModel(for: .anthropic)
        openAIKey       = KeychainStore.read("openai_api_key") ?? ""
        anthropicKey    = KeychainStore.read("anthropic_api_key") ?? ""
        // Default provider: Apple's free on-device model only when it would actually answer
        // right now. An Intel Mac (no Apple Intelligence, ever) or a Mac with it switched off
        // falls back to OpenAI rather than defaulting to a provider that errors on first use.
        let storedProvider = AIProvider(rawValue: defaults.string(forKey: "provider") ?? "")
        provider = storedProvider ?? (AppleIntelligence.isReadyNow ? .apple : .openai)
        // A stored choice of Apple is honoured while the hardware allows it (the model may just
        // be off or downloading), but dropped on hardware that can never run it.
        if provider == .apple, !AppleIntelligence.isDeviceCapable { provider = .openai }
        #if !os(macOS)
        engineMode = .remote
        #endif
    }

    var remoteURL: URL? {
        var s = remoteURLString.trimmingCharacters(in: .whitespacesAndNewlines)
        if s.isEmpty { return nil }
        if !s.contains("://") { s = "http://" + s }
        guard let url = URL(string: s), url.host != nil else { return nil }
        return url
    }

    // MARK: Keys / models per provider

    func key(for provider: AIProvider) -> String {
        switch provider {
        case .openai:    return openAIKey
        case .anthropic: return anthropicKey
        case .apple:     return ""
        }
    }

    func setKey(_ key: String, for provider: AIProvider) {
        switch provider {
        case .openai:    openAIKey = key
        case .anthropic: anthropicKey = key
        case .apple:     break
        }
    }

    /// Trimmed key for a provider, or nil when empty.
    func trimmedKey(for provider: AIProvider) -> String? {
        let k = key(for: provider).trimmingCharacters(in: .whitespacesAndNewlines)
        return k.isEmpty ? nil : k
    }

    var trimmedOpenAIKey: String? { trimmedKey(for: .openai) }
    var trimmedAnthropicKey: String? { trimmedKey(for: .anthropic) }

    /// Key for the selected provider (nil for Apple's on-device model, which needs none).
    var currentKey: String? { trimmedKey(for: provider) }

    var currentModel: String {
        switch provider {
        case .openai:    return openAIModel
        case .anthropic: return anthropicModel
        case .apple:     return "apple-on-device"
        }
    }

    func model(for provider: AIProvider) -> String {
        provider == .anthropic ? anthropicModel : openAIModel
    }

    func setModel(_ id: String, for provider: AIProvider) {
        if provider == .anthropic { anthropicModel = id } else if provider == .openai { openAIModel = id }
    }
}

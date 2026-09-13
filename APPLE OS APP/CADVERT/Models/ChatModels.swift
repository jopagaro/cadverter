import Foundation

struct ChatMessage: Identifiable, Equatable {
    enum Role: String { case user, assistant }

    let id: UUID
    var role: Role
    var text: String
    var isTyping: Bool = false
    var toolLabel: String? = nil
    var isError: Bool = false

    init(id: UUID = UUID(), role: Role, text: String, isTyping: Bool = false, toolLabel: String? = nil, isError: Bool = false) {
        self.id = id; self.role = role; self.text = text
        self.isTyping = isTyping; self.toolLabel = toolLabel; self.isError = isError
    }

    static func user(_ text: String) -> ChatMessage { ChatMessage(role: .user, text: text) }
    static func typing() -> ChatMessage { ChatMessage(role: .assistant, text: "", isTyping: true) }
}

/// One row of the transcript: the web UI mixes system pills, suggestion chips and bubbles.
enum ChatItem: Identifiable, Equatable {
    case system(id: UUID, text: String)
    case chips(id: UUID, questions: [String])
    case message(ChatMessage)

    var id: UUID {
        switch self {
        case .system(let id, _): return id
        case .chips(let id, _): return id
        case .message(let m): return m.id
        }
    }
}

/// Everything the app knows about the currently loaded part (one `/convert` result).
struct PartSession: Equatable {
    var id: String
    var filename: String
    var hsd: String
    var tier0: String?
    var images: [ViewImage]
    var format: String
    var isMesh: Bool
    var units: String
    var summary: PartSummary

    init(_ r: ConvertResponse) {
        id = r.sessionId; filename = r.filename; hsd = r.hsd; tier0 = r.tier0; images = r.images
        format = r.format; isMesh = r.isMesh; units = r.units; summary = r.summary
    }

    var stem: String { (filename as NSString).deletingPathExtension }

    /// The `.msg-system` line shown after a successful upload.
    var systemSummaryLine: String {
        if isMesh {
            return "\(filename) · \(format) (mesh) · Ask about shape and size."
        }
        var s = "\(filename) processed · \(summary.faces ?? 0) faces · \(summary.edges ?? 0) edges · \(units)"
        let feats = summary.sortedFeatures
        if !feats.isEmpty {
            s += " · " + feats.map { "\($0.count) \($0.name.lowercased().replacingOccurrences(of: "_", with: " "))" }.joined(separator: ", ")
        }
        return s
    }
}

/// Copied from index.html — the questions offered as chips after an upload.
enum SuggestedQuestions {
    static let all = [
        "What are the hole diameters?",
        "What is the overall bounding box?",
        "List all detected features",
        "Is this part machinable from one side?",
        "What is the thinnest wall?",
    ]
}

/// Copied from index.html `_toolLabel`.
enum ToolLabels {
    static let map: [String: String] = [
        "get_feature":       "Fetching feature geometry",
        "get_face":          "Reading face geometry",
        "get_edge":          "Reading edge geometry",
        "measure_distance":  "Computing exact distance",
        "get_neighbors":     "Exploring adjacent faces",
        "search_faces":      "Searching geometry",
        "get_cross_section": "Computing cross section",
    ]
    static func label(for tool: String) -> String { map[tool] ?? "Calling \(tool)" }
}

/// Copied from index.html `uploadFile` — the staged messages shown while the engine works.
enum ProcessingStages {
    static let stages: [(at: Int, label: String)] = [
        (2,   "Loading geometry…"),
        (8,   "Building topology graph…"),
        (30,  "Detecting features…"),
        (35,  "Computing spatial relationships…"),
        (90,  "Rendering views…"),
        (110, "Still working… (large assembly)"),
        (150, "Almost there…"),
    ]

    static func label(elapsed: Int) -> String {
        stages.reduce("Uploading file…") { elapsed >= $1.at ? $1.label : $0 }
    }

    static func percent(elapsed: Int) -> Double {
        min(95, 5 + Double(elapsed) * 0.9)
    }
}

struct ProcessingState: Equatable {
    var stage: String = "Uploading file…"
    var percent: Double = 5
    var elapsed: Int = 0
}

/// Who answers questions. Apple = Foundation Models on-device (macOS 26 / iOS 26); the other
/// two go through the server's `/chat` with an `X-Provider` header.
enum AIProvider: String, CaseIterable, Identifiable {
    case apple, openai, anthropic
    var id: String { rawValue }

    var label: String {
        switch self {
        case .apple:     return "Apple Intelligence"
        case .openai:    return "OpenAI"
        case .anthropic: return "Claude"
        }
    }

    var shortLabel: String {
        switch self {
        case .apple:     return "On-device"
        case .openai:    return "OpenAI"
        case .anthropic: return "Claude"
        }
    }

    /// Vendor name used in key prompts ("Add your Anthropic key").
    var keyVendor: String {
        switch self {
        case .apple:     return ""
        case .openai:    return "OpenAI"
        case .anthropic: return "Anthropic"
        }
    }

    var keyPlaceholder: String { self == .anthropic ? "sk-ant-..." : "sk-..." }

    var keyHint: String {
        switch self {
        case .apple:     return ""
        case .openai:    return "Get a key at platform.openai.com → API keys."
        case .anthropic: return "Get a key at console.anthropic.com → API keys."
        }
    }

    var needsKey: Bool { self != .apple }

    /// Providers worth offering on this Mac. Apple appears only on hardware that can run
    /// the on-device model — never on an Intel Mac, which ships macOS 26 but is excluded
    /// from Apple Intelligence. It still appears when the model is merely switched off or
    /// downloading, so the user can turn it on.
    static var selectable: [AIProvider] {
        (AppleIntelligence.isDeviceCapable ? [.apple] : []) + [.openai, .anthropic]
    }
}

/// Model lists per hosted provider, in the order shown in the pickers. Must stay within the
/// server's allow-lists (`ALLOWED_MODELS` / `ALLOWED_ANTHROPIC_MODELS` in server.py).
enum ChatModel {
    static let openAI: [(id: String, label: String)] = [
        ("gpt-4o-mini",  "GPT-4o mini"),
        ("gpt-4o",       "GPT-4o"),
        ("gpt-4.1",      "GPT-4.1"),
        ("gpt-4.1-mini", "GPT-4.1 mini"),
        ("gpt-5.4",      "GPT-5.4"),
        ("o4-mini",      "o4-mini"),
    ]
    static let anthropic: [(id: String, label: String)] = [
        ("claude-opus-5",    "Claude Opus 5"),
        ("claude-sonnet-5",  "Claude Sonnet 5"),
        ("claude-haiku-4-5", "Claude Haiku 4.5"),
        ("claude-opus-4-8",  "Claude Opus 4.8"),
        ("claude-sonnet-4-6","Claude Sonnet 4.6"),
    ]

    static func models(for provider: AIProvider) -> [(id: String, label: String)] {
        provider == .anthropic ? anthropic : openAI
    }

    static func defaultModel(for provider: AIProvider) -> String {
        provider == .anthropic ? "claude-opus-5" : "gpt-4o-mini"
    }

    static let `default` = "gpt-4o-mini"
}

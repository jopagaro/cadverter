import Foundation

// MARK: - Responses from cadvert-server (src/cadvert/server.py)

/// `GET /config`
struct ServerConfig: Decodable, Equatable {
    var disableAuth: Bool
    var stripeEnabled: Bool
    var stripeByokEnabled: Bool
    /// Per-provider capabilities (servers from 0.3.1 on); absent on older servers.
    var providers: [String: ProviderInfo]?
    var tools: [String]?

    enum CodingKeys: String, CodingKey {
        case disableAuth = "disable_auth"
        case stripeEnabled = "stripe_enabled"
        case stripeByokEnabled = "stripe_byok_enabled"
        case providers, tools
    }

    func info(for provider: AIProvider) -> ProviderInfo? { providers?[provider.rawValue] }
}

struct ProviderInfo: Decodable, Equatable {
    var available: Bool?
    var serverKey: Bool?
    var models: [String]?
    var defaultModel: String?

    enum CodingKeys: String, CodingKey {
        case available, models
        case serverKey = "server_key"
        case defaultModel = "default_model"
    }
}

/// One rendered orthographic / isometric view returned by `/convert`.
struct ViewImage: Decodable, Identifiable, Equatable {
    var name: String
    var url: String?
    var data: String?

    var id: String { name }

    /// Short label like the web strip shows ("front", "iso"…): strips the part stem prefix.
    var label: String {
        if let idx = name.lastIndex(of: "_") {
            let tail = name[name.index(after: idx)...]
            if !tail.isEmpty { return String(tail) }
        }
        return name
    }

    func resolvedURL(base: URL) -> URL? {
        if let url { return URL(string: url, relativeTo: base)?.absoluteURL }
        if let data { return URL(string: data) }
        return nil
    }
}

/// `summary` block of `/convert` (see `_build_summary`).
struct PartSummary: Decodable, Equatable {
    var format: String?
    var schema: String?
    var units: String?
    var isMesh: Bool?
    var faces: Int?
    var edges: Int?
    var bodies: Int?
    var triangles: Int?
    var gdtCount: Int?
    var features: [String: Int]?

    enum CodingKeys: String, CodingKey {
        case format, schema, units, faces, edges, bodies, triangles, features
        case isMesh = "is_mesh"
        case gdtCount = "gdt_count"
    }

    /// Feature rows in a stable order (most common first, then alphabetical).
    var sortedFeatures: [(name: String, count: Int)] {
        (features ?? [:])
            .map { (name: $0.key, count: $0.value) }
            .sorted { $0.count != $1.count ? $0.count > $1.count : $0.name < $1.name }
    }
}

/// `POST /convert`
struct ConvertResponse: Decodable, Equatable {
    var sessionId: String
    var hsd: String
    /// Compact Tier-0 summary (the LLM system-prompt context). Absent on older servers.
    var tier0: String?
    var images: [ViewImage]
    var format: String
    var isMesh: Bool
    var filename: String
    var units: String
    var summary: PartSummary

    enum CodingKeys: String, CodingKey {
        case hsd, tier0, images, format, filename, units, summary
        case sessionId = "session_id"
        case isMesh = "is_mesh"
    }
}

/// FastAPI error bodies: `{"detail": "..."}` or `{"detail": {"error": "...", "message": "..."}}`.
struct APIErrorBody {
    var message: String
    var code: String?
    var messagesUsed: Int?

    init?(data: Data) {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        if let s = obj["detail"] as? String {
            message = s; code = nil; messagesUsed = nil
        } else if let d = obj["detail"] as? [String: Any] {
            message = (d["message"] as? String) ?? (d["error"] as? String) ?? "Request failed"
            code = d["error"] as? String
            messagesUsed = d["messages_used"] as? Int
        } else if let e = obj["error"] as? String {
            message = e; code = nil; messagesUsed = nil
        } else {
            return nil
        }
    }
}

// MARK: - Errors

enum CadvertError: LocalizedError, Equatable {
    case unauthorized(String)
    case fileLimit(String)
    case byokRequired(String)
    case byokKeyRequired(String)
    case sessionLimit(String)
    case serverKeyMissing(String)
    case http(status: Int, message: String)
    case unsupportedFormat(String)
    case notConnected
    case invalidResponse
    case stream(String)
    case engine(String)

    var errorDescription: String? {
        switch self {
        case .unauthorized(let m), .fileLimit(let m), .byokRequired(let m), .byokKeyRequired(let m),
             .sessionLimit(let m), .serverKeyMissing(let m), .stream(let m), .engine(let m):
            return m
        case .http(_, let m): return m
        case .unsupportedFormat(let ext):
            return "Unsupported file type .\(ext) — supported: \(SupportedFormat.allExtensions.joined(separator: ", "))"
        case .notConnected: return "Not connected to a CADVERT engine — check Settings"
        case .invalidResponse: return "Unexpected response from server"
        }
    }

    /// Map an HTTP failure the way `index.html` does.
    static func from(status: Int, data: Data) -> CadvertError {
        let body = APIErrorBody(data: data)
        let msg = body?.message ?? HTTPURLResponse.localizedString(forStatusCode: status)
        switch (status, body?.code) {
        case (401, _):                    return .unauthorized(msg)
        case (429, "byok_required"):      return .byokRequired(msg)
        case (429, "byok_key_required"):  return .byokKeyRequired(msg)
        case (429, "session_limit"):      return .sessionLimit(msg)
        case (429, _):                    return .fileLimit(msg)
        case (503, _):                    return .serverKeyMissing(msg)
        default:                          return .http(status: status, message: msg)
        }
    }
}

// MARK: - Streaming chat events (`data: {...}` lines from `/chat/{session}`)

enum ChatEvent: Equatable {
    case content(String)
    case toolCall(String)
    case error(String)
    case done
}

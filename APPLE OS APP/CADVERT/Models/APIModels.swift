import Foundation

// MARK: - Responses from cadvert-server (src/cadvert/server.py)

/// `GET /config`
struct ServerConfig: Decodable, Equatable {
    /// Every field is optional: the engine is versioned independently of the app, and a
    /// missing key must degrade rather than fail the whole connection.
    var localOnly: Bool?
    var providers: [String: ProviderInfo]?
    var tools: [String]?
    var maxFileMB: Int?

    enum CodingKeys: String, CodingKey {
        case providers, tools
        case localOnly = "local_only"
        case maxFileMB = "max_file_mb"
    }

    func info(for provider: AIProvider) -> ProviderInfo? { providers?[provider.rawValue] }
}

struct ProviderInfo: Decodable, Equatable {
    var available: Bool?
    var keyPresent: Bool?
    var models: [String]?
    var defaultModel: String?

    enum CodingKeys: String, CodingKey {
        case available, models
        case keyPresent = "key_present"
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
    case apiKeyRequired(String)
    case http(status: Int, message: String)
    case unsupportedFormat(String)
    case notConnected
    case invalidResponse
    case stream(String)
    case engine(String)

    var errorDescription: String? {
        switch self {
        case .unauthorized(let m), .fileLimit(let m),
             .apiKeyRequired(let m), .stream(let m), .engine(let m):
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
        case (_, "api_key_required"):     return .apiKeyRequired(msg)
        case (401, _):                    return .unauthorized(msg)
        case (429, _):                    return .fileLimit(msg)
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


/// `GET /cache` — what the engine is holding on disk.
struct CacheUsage: Decodable, Equatable {
    var path: String?
    var sessions: Int
    var files: Int
    var megabytes: Double
    var ttlHours: Int?

    enum CodingKeys: String, CodingKey {
        case path, sessions, files, megabytes
        case ttlHours = "ttl_hours"
    }

    /// "26.4 MB across 15 parts", or a plain empty state.
    var summary: String {
        guard sessions > 0 else { return "No cached parts" }
        let size = megabytes >= 1024
            ? String(format: "%.1f GB", megabytes / 1024)
            : String(format: "%.0f MB", megabytes)
        return "\(size) across \(sessions) part\(sessions == 1 ? "" : "s")"
    }
}

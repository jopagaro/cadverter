import Foundation

/// Thin client for the cadvert REST API (`src/cadvert/server.py`).
/// Works against the bundled local engine or any remote `cadvert-server`.
struct CadvertClient {
    var baseURL: URL

    /// Long timeouts: `/convert` can legitimately take minutes on large assemblies (server cap is 10 min).
    static let session: URLSession = {
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = 660
        cfg.timeoutIntervalForResource = 3600
        // Never wait for connectivity: while the local engine is still booting, a request must fail fast
        // (connection refused) so the readiness poll can retry instead of hanging.
        cfg.waitsForConnectivity = false
        return URLSession(configuration: cfg)
    }()

    // MARK: Requests

    private func request(_ path: String, method: String = "GET") -> URLRequest {
        var req = URLRequest(url: URL(string: path, relativeTo: baseURL)!.absoluteURL)
        req.httpMethod = method
        return req
    }

    /// `GET /config`
    func fetchConfig() async throws -> ServerConfig {
        var req = request("/config")
        req.timeoutInterval = 10
        let (data, resp) = try await Self.session.data(for: req)
        try Self.check(resp, data)
        return try JSONDecoder().decode(ServerConfig.self, from: data)
    }

    /// `POST /convert` — multipart upload of one CAD file.
    func convert(fileData: Data, filename: String) async throws -> ConvertResponse {
        var req = request("/convert", method: "POST")
        let boundary = "cadvert-\(UUID().uuidString)"
        req.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()
        body.append("--\(boundary)\r\n")
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"\(Self.sanitize(filename))\"\r\n")
        body.append("Content-Type: application/octet-stream\r\n\r\n")
        body.append(fileData)
        body.append("\r\n--\(boundary)--\r\n")

        let (data, resp) = try await Self.session.upload(for: req, from: body)
        try Self.check(resp, data)
        do {
            return try JSONDecoder().decode(ConvertResponse.self, from: data)
        } catch {
            throw CadvertError.invalidResponse
        }
    }

    /// Builds the `/chat` request — provider, model and BYOK key travel as headers.
    func chatRequest(sessionId: String, history: [[String: String]], provider: AIProvider, apiKey: String?, model: String) throws -> URLRequest {
        var req = request("/chat/\(sessionId)", method: "POST")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        req.setValue(provider == .anthropic ? "anthropic" : "openai", forHTTPHeaderField: "X-Provider")
        req.setValue(model, forHTTPHeaderField: "X-Model")
        if let apiKey, !apiKey.isEmpty {
            req.setValue(apiKey, forHTTPHeaderField: provider == .anthropic ? "X-Anthropic-Key" : "X-OpenAI-Key")
        }
        req.httpBody = try JSONSerialization.data(withJSONObject: ["messages": history])
        return req
    }

    /// `POST /chat/{session}` — streams `ChatEvent`s until `[DONE]`.
    func chat(sessionId: String, history: [[String: String]], provider: AIProvider, apiKey: String?, model: String) -> AsyncThrowingStream<ChatEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let req = try chatRequest(sessionId: sessionId, history: history, provider: provider, apiKey: apiKey, model: model)

                    let (bytes, resp) = try await Self.session.bytes(for: req)
                    guard let http = resp as? HTTPURLResponse else { throw CadvertError.invalidResponse }
                    if http.statusCode != 200 {
                        var errData = Data()
                        for try await b in bytes { errData.append(b) }
                        throw CadvertError.from(status: http.statusCode, data: errData)
                    }
                    for try await line in bytes.lines {
                        if Task.isCancelled { break }
                        guard let event = SSEParser.parse(line: line) else { continue }
                        switch event {
                        case .done:
                            continuation.yield(.done)
                            continuation.finish()
                            return
                        case .error(let msg):
                            throw CadvertError.stream(msg)
                        default:
                            continuation.yield(event)
                        }
                    }
                    continuation.yield(.done)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// `POST /tool/{session}` — run one geometry tool; returns the JSON result as text
    /// (what a model wants to read). Used by the Apple on-device provider.
    func executeTool(sessionId: String, name: String, arguments: [String: Any]) async throws -> String {
        var req = request("/tool/\(sessionId)", method: "POST")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.timeoutInterval = 120
        req.httpBody = try JSONSerialization.data(withJSONObject: ["name": name, "arguments": arguments])
        let (data, resp) = try await Self.session.data(for: req)
        try Self.check(resp, data)
        return String(decoding: data, as: UTF8.self)
    }

    /// `GET /cache` — how much disk the cached parts use.
    func cacheUsage() async throws -> CacheUsage {
        var req = request("/cache")
        req.timeoutInterval = 30
        let (data, resp) = try await Self.session.data(for: req)
        try Self.check(resp, data)
        return try JSONDecoder().decode(CacheUsage.self, from: data)
    }

    /// `DELETE /cache` — remove every cached part. Safe: all of it rebuilds from the file.
    @discardableResult
    func clearCache() async throws -> CacheUsage {
        var req = request("/cache", method: "DELETE")
        req.timeoutInterval = 60
        let (data, resp) = try await Self.session.data(for: req)
        try Self.check(resp, data)
        struct Cleared: Decodable { let now: CacheUsage }
        return try JSONDecoder().decode(Cleared.self, from: data).now
    }

    /// `DELETE /session/{id}` — best effort cleanup.
    func deleteSession(_ id: String) async {
        var req = request("/session/\(id)", method: "DELETE")
        req.timeoutInterval = 10
        _ = try? await Self.session.data(for: req)
    }

    func imageURL(for image: ViewImage) -> URL? { image.resolvedURL(base: baseURL) }

    // MARK: Helpers

    private static func check(_ resp: URLResponse, _ data: Data) throws {
        guard let http = resp as? HTTPURLResponse else { throw CadvertError.invalidResponse }
        guard (200..<300).contains(http.statusCode) else {
            throw CadvertError.from(status: http.statusCode, data: data)
        }
    }

    private static func sanitize(_ name: String) -> String {
        name.replacingOccurrences(of: "\"", with: "_").replacingOccurrences(of: "\r", with: "").replacingOccurrences(of: "\n", with: "")
    }
}

private extension Data {
    mutating func append(_ s: String) { append(s.data(using: .utf8)!) }
}

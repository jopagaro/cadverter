import Foundation

/// Parses the server-sent-event lines emitted by `POST /chat/{session}`:
///
///     data: {"content": "…"}
///     data: {"tool_call": "get_feature"}
///     data: {"highlight": {"faces": [12, 30], "label": "Thinnest wall · 1.6 mm"}}
///     data: {"error": "…"}
///     data: [DONE]
///
/// Mirrors the reader loop in `index.html` (`_sendText`).
enum SSEParser {
    static func parse(line: String) -> ChatEvent? {
        let trimmed = line.trimmingCharacters(in: .newlines)
        guard trimmed.hasPrefix("data: ") else { return nil }
        let payload = String(trimmed.dropFirst(6))
        if payload == "[DONE]" { return .done }
        guard let data = payload.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil   // partial / malformed JSON — the web client ignores these too
        }
        if let err = obj["error"] as? String { return .error(err) }
        if let tool = obj["tool_call"] as? String { return .toolCall(tool) }
        if let h = obj["highlight"] as? [String: Any], let faces = h["faces"] as? [Int], !faces.isEmpty {
            return .highlight(ViewerHighlight(faces: faces, label: h["label"] as? String ?? ""))
        }
        if let content = obj["content"] as? String { return .content(content) }
        return nil
    }
}

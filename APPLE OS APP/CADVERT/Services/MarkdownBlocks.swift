import Foundation

/// The assistant's replies are markdown-ish; the web UI has a tiny renderer (`renderMarkdown`).
/// We split the text into blocks and let SwiftUI's inline markdown handle emphasis and `code`.
enum MarkdownBlock: Equatable {
    case paragraph(String)
    case code(String)
    case list([String])
}

enum MarkdownBlocks {
    static func parse(_ text: String) -> [MarkdownBlock] {
        var blocks: [MarkdownBlock] = []
        // 1. Split on ``` fences (odd segments are code).
        let segments = text.components(separatedBy: "```")
        for (i, seg) in segments.enumerated() {
            if i % 2 == 1 {
                var code = seg
                // Drop an optional language hint on the first line ("```python\n")
                if let nl = code.firstIndex(of: "\n") {
                    let first = code[..<nl]
                    if !first.contains(" "), first.count <= 20 { code = String(code[code.index(after: nl)...]) }
                }
                let trimmed = code.trimmingCharacters(in: .newlines)
                if !trimmed.isEmpty { blocks.append(.code(trimmed)) }
            } else {
                blocks.append(contentsOf: parseProse(seg))
            }
        }
        return blocks
    }

    private static func parseProse(_ text: String) -> [MarkdownBlock] {
        var out: [MarkdownBlock] = []
        var para: [String] = []
        var list: [String] = []

        func flushPara() {
            let joined = para.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
            if !joined.isEmpty { out.append(.paragraph(joined)) }
            para.removeAll()
        }
        func flushList() {
            if !list.isEmpty { out.append(.list(list)) }
            list.removeAll()
        }

        for rawLine in text.components(separatedBy: "\n") {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if isBullet(rawLine) {
                flushPara()
                list.append(stripBullet(line))
            } else if line.isEmpty {
                flushPara(); flushList()
            } else {
                flushList()
                para.append(rawLine)
            }
        }
        flushPara(); flushList()
        return out
    }

    private static func isBullet(_ line: String) -> Bool {
        let t = line.trimmingCharacters(in: .whitespaces)
        if t.hasPrefix("- ") || t.hasPrefix("* ") || t.hasPrefix("• ") { return true }
        // "1. item"
        if let dot = t.firstIndex(of: "."), dot != t.startIndex, t[..<dot].allSatisfy(\.isNumber),
           t.index(after: dot) < t.endIndex, t[t.index(after: dot)] == " " { return true }
        return false
    }

    private static func stripBullet(_ line: String) -> String {
        var s = line
        if s.hasPrefix("- ") || s.hasPrefix("* ") || s.hasPrefix("• ") { s.removeFirst(2) }
        else if let dot = s.firstIndex(of: "."), s[..<dot].allSatisfy(\.isNumber) {
            s = String(s[s.index(after: dot)...]).trimmingCharacters(in: .whitespaces)
        }
        return s.trimmingCharacters(in: .whitespaces)
    }
}

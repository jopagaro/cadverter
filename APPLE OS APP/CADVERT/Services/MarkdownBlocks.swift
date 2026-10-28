import Foundation

/// The assistant writes markdown. SwiftUI only styles *inline* markdown (bold, `code`, links),
/// so replies are split into blocks here and each block gets a native view: headings, lists,
/// tables, code, and ```chart blocks drawn with Swift Charts.
///
/// Parsing is line-based and tolerant of a reply that is still streaming in: a table whose
/// separator row hasn't arrived reads as a paragraph for a moment, and an unclosed ```chart
/// fence shows a placeholder instead of half a JSON object.
enum MarkdownBlock: Equatable {
    case heading(level: Int, text: String)
    case paragraph(String)
    case list([ListItem])
    case table(Table)
    case code(String)
    case chart(ChartSpec)
    case chartPending
    case quote(String)
    case rule

    struct ListItem: Equatable {
        var marker: String?      // nil for a bullet, "1." for a numbered item
        var text: String
        var depth: Int = 0
    }

    struct Table: Equatable {
        var header: [String]
        var rows: [[String]]
        var alignments: [Alignment]
    }

    enum Alignment: Equatable { case leading, center, trailing }
}

enum MarkdownBlocks {
    static func parse(_ text: String) -> [MarkdownBlock] {
        var blocks: [MarkdownBlock] = []
        var prose: [String] = []
        var fence: (info: String, lines: [String])?

        for line in text.components(separatedBy: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("```") {
                if let open = fence {
                    blocks.append(contentsOf: fenced(info: open.info, body: open.lines, closed: true))
                    fence = nil
                } else {
                    blocks.append(contentsOf: parseProse(prose)); prose.removeAll()
                    fence = (String(trimmed.dropFirst(3)).trimmingCharacters(in: .whitespaces).lowercased(), [])
                }
            } else if fence != nil {
                fence!.lines.append(line)
            } else {
                prose.append(line)
            }
        }
        blocks.append(contentsOf: parseProse(prose))
        if let open = fence { blocks.append(contentsOf: fenced(info: open.info, body: open.lines, closed: false)) }
        return blocks
    }

    // MARK: Fenced blocks

    private static func fenced(info: String, body: [String], closed: Bool) -> [MarkdownBlock] {
        let content = body.joined(separator: "\n").trimmingCharacters(in: .newlines)
        if info == "chart" {
            if !closed { return [.chartPending] }
            if let spec = ChartSpec.decode(content) { return [.chart(spec)] }
        }
        return content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? [] : [.code(content)]
    }

    // MARK: Prose

    private static func parseProse(_ lines: [String]) -> [MarkdownBlock] {
        var out: [MarkdownBlock] = []
        var para: [String] = []
        var list: [MarkdownBlock.ListItem] = []
        var quote: [String] = []

        func flushPara() {
            let joined = para.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
            if !joined.isEmpty { out.append(.paragraph(joined)) }
            para.removeAll()
        }
        func flushList() { if !list.isEmpty { out.append(.list(list)) }; list.removeAll() }
        func flushQuote() {
            let joined = quote.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
            if !joined.isEmpty { out.append(.quote(joined)) }
            quote.removeAll()
        }
        func flushAll() { flushPara(); flushList(); flushQuote() }

        var i = 0
        while i < lines.count {
            let line = lines[i]
            let t = line.trimmingCharacters(in: .whitespaces)

            if t.isEmpty { flushAll(); i += 1; continue }

            if let h = heading(t) {
                flushAll(); out.append(h); i += 1; continue
            }
            if i + 1 < lines.count, t.contains("|"), isTableSeparator(lines[i + 1]) {
                flushAll()
                let header = cells(t)
                let aligns = alignments(lines[i + 1], count: header.count)
                var rows: [[String]] = []
                i += 2
                while i < lines.count {
                    let r = lines[i].trimmingCharacters(in: .whitespaces)
                    guard !r.isEmpty, r.contains("|") else { break }
                    rows.append(fit(cells(r), header.count)); i += 1
                }
                out.append(.table(.init(header: header, rows: rows, alignments: inferNumeric(aligns, rows))))
                continue
            }
            if isRule(t) { flushAll(); out.append(.rule); i += 1; continue }
            if t.hasPrefix(">") {
                flushPara(); flushList()
                quote.append(String(t.dropFirst()).trimmingCharacters(in: .whitespaces)); i += 1; continue
            }
            if let item = listItem(line) {
                flushPara(); flushQuote()
                list.append(item); i += 1; continue
            }
            // An indented line right after a list item continues that item.
            if !list.isEmpty, line.hasPrefix("  ") {
                list[list.count - 1].text += " " + t; i += 1; continue
            }
            flushList(); flushQuote()
            para.append(line)
            i += 1
        }
        flushAll()
        return out
    }

    private static func heading(_ t: String) -> MarkdownBlock? {
        let hashes = t.prefix { $0 == "#" }.count
        guard (1...6).contains(hashes), t.dropFirst(hashes).first == " " else { return nil }
        var text = t.dropFirst(hashes).trimmingCharacters(in: .whitespaces)
        while text.hasSuffix("#") { text.removeLast() }
        text = text.trimmingCharacters(in: .whitespaces)
        return text.isEmpty ? nil : .heading(level: hashes, text: text)
    }

    private static func isRule(_ t: String) -> Bool {
        let s = t.replacingOccurrences(of: " ", with: "")
        guard s.count >= 3, let c = s.first, "-*_".contains(c) else { return false }
        return s.allSatisfy { $0 == c }
    }

    private static func listItem(_ line: String) -> MarkdownBlock.ListItem? {
        let indent = line.prefix { $0 == " " || $0 == "\t" }.reduce(0) { $0 + ($1 == "\t" ? 4 : 1) }
        let t = line.trimmingCharacters(in: .whitespaces)
        let depth = min(indent / 2, 3)
        for b in ["- ", "* ", "• ", "+ "] where t.hasPrefix(b) {
            let rest = String(t.dropFirst(2)).trimmingCharacters(in: .whitespaces)
            return rest.isEmpty ? nil : .init(marker: nil, text: rest, depth: depth)
        }
        let digits = t.prefix { $0.isNumber }
        guard !digits.isEmpty, digits.count <= 3 else { return nil }
        let after = t.dropFirst(digits.count)
        guard let p = after.first, p == "." || p == ")", after.dropFirst().first == " " else { return nil }
        let rest = String(after.dropFirst(2)).trimmingCharacters(in: .whitespaces)
        return rest.isEmpty ? nil : .init(marker: "\(digits).", text: rest, depth: depth)
    }

    // MARK: Tables

    private static func isTableSeparator(_ line: String) -> Bool {
        let t = line.trimmingCharacters(in: .whitespaces)
        guard t.contains("-"), t.contains("|") || t.hasPrefix(":") || t.hasPrefix("-") else { return false }
        let parts = cells(t)
        return !parts.isEmpty && parts.allSatisfy { cell in
            let c = cell.trimmingCharacters(in: .whitespaces)
            return !c.isEmpty && c.allSatisfy { $0 == "-" || $0 == ":" } && c.contains("-")
        }
    }

    /// `| a | b |` → ["a", "b"]; `\|` stays a literal pipe inside a cell.
    private static func cells(_ row: String) -> [String] {
        var s = row.trimmingCharacters(in: .whitespaces)
        if s.hasPrefix("|") { s.removeFirst() }
        if s.hasSuffix("|") && !s.hasSuffix("\\|") { s.removeLast() }
        var out: [String] = [], cur = "", escaped = false
        for ch in s {
            if escaped { cur.append(ch); escaped = false; continue }
            if ch == "\\" { escaped = true; continue }
            if ch == "|" { out.append(cur.trimmingCharacters(in: .whitespaces)); cur = "" } else { cur.append(ch) }
        }
        out.append(cur.trimmingCharacters(in: .whitespaces))
        return out
    }

    private static func alignments(_ sep: String, count: Int) -> [MarkdownBlock.Alignment] {
        fit(cells(sep), count).map { c in
            let c = c.trimmingCharacters(in: .whitespaces)
            switch (c.hasPrefix(":"), c.hasSuffix(":")) {
            case (true, true): return .center
            case (false, true): return .trailing
            default: return .leading
            }
        }
    }

    /// Columns the model left unaligned but that hold only numbers ("42", "∅6.600 mm", "12×",
    /// "77%") read best right-aligned, so their digits line up. A cell that merely *starts* with
    /// a quantity ("2× 12 V motors") is text and stays left-aligned.
    private static func inferNumeric(_ aligns: [MarkdownBlock.Alignment], _ rows: [[String]]) -> [MarkdownBlock.Alignment] {
        aligns.enumerated().map { col, a in
            guard a == .leading, col > 0 else { return a }
            let values = rows.map { $0[col] }.filter { !$0.isEmpty && $0 != "—" && $0 != "-" }
            guard !values.isEmpty else { return a }
            let numeric = values.filter { isNumberCell($0) }
            return Double(numeric.count) >= Double(values.count) * 0.8 ? .trailing : a
        }
    }

    private static let numberCell = try! NSRegularExpression(
        pattern: #"^[∅⌀Ø~≈±<>≤≥]?\s*[-+]?\d[\d,]*(\.\d+)?\s*(×|x|%|°|[A-Za-zµ²³/]{1,4})?$"#)

    static func isNumberCell(_ s: String) -> Bool {
        let t = s.replacingOccurrences(of: "*", with: "").trimmingCharacters(in: .whitespaces)
        return numberCell.firstMatch(in: t, range: NSRange(t.startIndex..., in: t)) != nil
    }

    private static func fit<T>(_ a: [T], _ n: Int, pad: T) -> [T] {
        a.count >= n ? Array(a.prefix(n)) : a + Array(repeating: pad, count: n - a.count)
    }
    private static func fit(_ a: [String], _ n: Int) -> [String] { fit(a, n, pad: "") }
}

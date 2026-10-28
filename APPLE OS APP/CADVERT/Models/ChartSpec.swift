import Foundation

/// A chart the assistant asked for with a fenced ```chart block of JSON:
///
///     {"type": "bar", "title": "Holes by diameter", "x": "Diameter (mm)", "y": "Count",
///      "labels": ["4.1", "4.9"], "series": [{"name": "Blind", "values": [42, 42]}]}
///
/// Decoding is deliberately forgiving: models vary in how they spell the same idea, so a
/// single `values` array, a `data: [{label, value}]` list and the usual type aliases all work.
struct ChartSpec: Equatable {
    enum Kind: String, Equatable {
        case bar       // horizontal bars: named categories, long labels
        case column    // vertical bars
        case line      // values along an ordered x
        case donut     // parts of a whole
    }

    struct Series: Equatable {
        var name: String
        var values: [Double]       // .nan where the model gave no number
    }

    var kind: Kind
    var title: String?
    var xLabel: String?
    var yLabel: String?
    var labels: [String]
    var series: [Series]

    /// Colours are validated for at most six series side by side, and a donut reads at
    /// six slices; anything beyond folds into "Other" rather than inventing a seventh hue.
    static let maxSeries = 6
    static let maxLabels = 24

    static func decode(_ json: String) -> ChartSpec? {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        return ChartSpec(json: obj)?.normalized()
    }

    init(kind: Kind, title: String? = nil, xLabel: String? = nil, yLabel: String? = nil,
         labels: [String], series: [Series]) {
        self.kind = kind; self.title = title; self.xLabel = xLabel; self.yLabel = yLabel
        self.labels = labels; self.series = series
    }

    private init?(json o: [String: Any]) {
        let rawKind = ((o["type"] ?? o["kind"] ?? o["chart"]) as? String ?? "bar").lowercased()
        switch rawKind {
        case "bar", "hbar", "barh", "horizontal", "horizontal_bar", "horizontal-bar": kind = .bar
        case "column", "columns", "vbar", "vertical", "vertical_bar", "histogram": kind = .column
        case "line", "lines", "trend": kind = .line
        case "donut", "doughnut", "pie", "ring": kind = .donut
        default: return nil
        }
        title = Self.string(o["title"])
        xLabel = Self.string(o["x"] ?? o["x_label"] ?? o["xLabel"] ?? o["x_axis"])
        yLabel = Self.string(o["y"] ?? o["y_label"] ?? o["yLabel"] ?? o["y_axis"] ?? o["unit"])

        if let rows = o["data"] as? [[String: Any]] {                 // [{label, value}]
            labels = rows.map { Self.string($0["label"] ?? $0["name"] ?? $0["x"]) ?? "" }
            series = [Series(name: yLabel ?? "Value",
                             values: rows.map { Self.number($0["value"] ?? $0["y"] ?? $0["count"]) })]
        } else {
            labels = (o["labels"] as? [Any] ?? o["categories"] as? [Any] ?? []).map { Self.string($0) ?? "" }
            if let list = o["series"] as? [[String: Any]] {
                series = list.enumerated().map { i, s in
                    Series(name: Self.string(s["name"] ?? s["label"]) ?? "Series \(i + 1)",
                           values: (s["values"] as? [Any] ?? s["data"] as? [Any] ?? []).map(Self.number))
                }
            } else if let values = o["values"] as? [Any] {
                series = [Series(name: yLabel ?? "Value", values: values.map(Self.number))]
            } else {
                series = []
            }
        }
    }

    /// Clamps to what the renderer can show legibly. Nil when nothing plottable is left.
    func normalized() -> ChartSpec? {
        var s = self
        s.labels = Array(s.labels.prefix(Self.maxLabels))
        let n = s.labels.count
        guard n > 0 else { return nil }
        s.series = s.series.map { ser in
            var v = Array(ser.values.prefix(n))
            if v.count < n { v += Array(repeating: .nan, count: n - v.count) }
            return Series(name: ser.name, values: v)
        }.filter { $0.values.contains { !$0.isNaN } }
        guard !s.series.isEmpty else { return nil }

        if s.kind == .donut {
            // One whole, split by label. Keep the largest five and fold the rest into "Other".
            let values = s.series[0].values.map { $0.isNaN ? 0 : max(0, $0) }
            var pairs = zip(s.labels, values).filter { $0.1 > 0 }
            guard !pairs.isEmpty else { return nil }
            if pairs.count > Self.maxSeries {
                pairs.sort { $0.1 > $1.1 }
                let other = pairs[(Self.maxSeries - 1)...].reduce(0) { $0 + $1.1 }
                pairs = Array(pairs[..<(Self.maxSeries - 1)]) + [("Other", other)]
            }
            s.labels = pairs.map(\.0)
            s.series = [Series(name: s.series[0].name, values: pairs.map(\.1))]
        } else if s.series.count > Self.maxSeries {
            s.series = Array(s.series.prefix(Self.maxSeries))
        }
        return s
    }

    /// The same numbers as a table — the "Show data" view under every chart.
    var table: MarkdownBlock.Table {
        let header = [xLabel ?? (kind == .donut ? "Item" : "")] + series.map(\.name)
        let rows = labels.indices.map { i in [labels[i]] + series.map { Self.format($0.values[i]) } }
        return MarkdownBlock.Table(header: header, rows: rows,
                                   alignments: [.leading] + Array(repeating: .trailing, count: series.count))
    }

    /// 42 → "42", 6.6 → "6.6", 0.12345 → "0.123"; no trailing zeros.
    static func format(_ v: Double) -> String {
        guard !v.isNaN else { return "–" }
        if v == v.rounded(), abs(v) < 1e15 { return String(Int64(v)) }
        let s = String(format: abs(v) >= 100 ? "%.1f" : "%.3g", v)
        return s.contains(".") && !s.contains("e") ? s.replacingOccurrences(of: #"0+$"#, with: "", options: .regularExpression)
            .replacingOccurrences(of: #"\.$"#, with: "", options: .regularExpression) : s
    }

    private static func string(_ v: Any?) -> String? {
        switch v {
        case let s as String: return s.isEmpty ? nil : s
        case let n as NSNumber: return format(n.doubleValue)
        default: return nil
        }
    }

    private static func number(_ v: Any?) -> Double {
        switch v {
        case let n as NSNumber: return n.doubleValue
        case let s as String:
            let cleaned = s.filter { "0123456789.-eE".contains($0) }
            return Double(cleaned) ?? .nan
        default: return .nan
        }
    }
}

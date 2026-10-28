import SwiftUI
import Charts
#if canImport(AppKit)
import AppKit
#else
import UIKit
#endif

/// Renders an assistant reply: headings, paragraphs (inline **bold**, *em*, `code`), lists,
/// tables, quotes, code blocks and ```chart blocks.
struct MarkdownText: View {
    var text: String
    var isUser: Bool
    @Environment(\.palette) private var p

    var body: some View {
        let blocks = MarkdownBlocks.parse(text)
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { index, block in
                view(for: block, first: index == 0)
            }
        }
    }

    private var ink: Color { isUser ? .white : p.text }

    @ViewBuilder
    private func view(for block: MarkdownBlock, first: Bool) -> some View {
        switch block {
        case .heading(let level, let s):
            Text(InlineMarkdown.attributed(s, isUser: isUser, palette: p))
                .typo(level == 1 ? 15 : level == 2 ? 14 : 12.5, .semibold)
                .foregroundStyle(ink)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, first ? 0 : 4)
        case .paragraph(let s):
            Text(InlineMarkdown.attributed(s, isUser: isUser, palette: p))
                .typo(12.5, .regular)
                .lineSpacing(3)
                .foregroundStyle(ink)
                .fixedSize(horizontal: false, vertical: true)
        case .list(let items):
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        Text(item.marker ?? (item.depth == 0 ? "•" : "◦"))
                            .typo(12.5, item.marker == nil ? .medium : .semibold)
                            .monospacedDigit()
                            .foregroundStyle(isUser ? .white : p.textMuted)
                            .frame(minWidth: item.marker == nil ? 8 : 16, alignment: .trailing)
                        Text(InlineMarkdown.attributed(item.text, isUser: isUser, palette: p))
                            .typo(12.5, .regular)
                            .lineSpacing(3)
                            .foregroundStyle(ink)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .padding(.leading, CGFloat(item.depth) * 16)
                }
            }
            .padding(.leading, 2)
        case .table(let table):
            TableBlockView(table: table)
        case .code(let code):
            ScrollView(.horizontal, showsIndicators: false) {
                Text(code)
                    .typo(11, .regular, mono: true)
                    .foregroundStyle(ink)
                    .padding(12)
            }
            .background(isUser ? Color.white.opacity(0.15) : p.bg, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
        case .chart(let spec):
            ChartBlockView(spec: spec)
        case .chartPending:
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text("Drawing chart…").typo(11, .medium).foregroundStyle(p.textMuted)
            }
            .frame(maxWidth: .infinity, minHeight: 80)
            .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).stroke(p.border, lineWidth: 1))
        case .quote(let s):
            HStack(alignment: .top, spacing: 10) {
                RoundedRectangle(cornerRadius: 1.5).fill(p.border).frame(width: 3)
                Text(InlineMarkdown.attributed(s, isUser: isUser, palette: p))
                    .typo(12.5, .regular)
                    .lineSpacing(3)
                    .foregroundStyle(isUser ? .white.opacity(0.85) : p.textMuted)
                    .fixedSize(horizontal: false, vertical: true)
            }
        case .rule:
            Rectangle().fill(p.border).frame(height: 1).padding(.vertical, 2)
        }
    }
}

/// SwiftUI's inline markdown, plus the bubble's style for `code` spans.
enum InlineMarkdown {
    static func attributed(_ s: String, isUser: Bool, palette p: Palette, size: CGFloat = 10.5) -> AttributedString {
        var a = (try? AttributedString(markdown: s, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)))
            ?? AttributedString(s)
        for run in a.runs where run.inlinePresentationIntent?.contains(.code) == true {
            a[run.range].font = Typo.font(size, .medium, mono: true)
            a[run.range].backgroundColor = isUser ? Color.white.opacity(0.2) : p.surfaceAlt
        }
        return a
    }
}

// MARK: - Tables

/// A markdown table as a real grid: header row, hairline rules, numbers right-aligned with
/// tabular digits. Wide tables scroll sideways inside the bubble instead of wrapping into mush.
struct TableBlockView: View {
    var table: MarkdownBlock.Table
    @Environment(\.palette) private var p

    private static let fontSize: CGFloat = 11.5
    private static let maxColumn: CGFloat = 240     // wider text wraps inside its column
    private static let padH: CGFloat = 12
    private static let padV: CGFloat = 8

    var body: some View {
        let widths = columnWidths()
        // Fits the bubble → shown as is; wider → scrolls sideways rather than squashing.
        ViewThatFits(in: .horizontal) {
            framed(grid(widths))
            framed(ScrollView(.horizontal, showsIndicators: false) { grid(widths) })
        }
    }

    private func framed(_ content: some View) -> some View {
        content
            .background(p.surface)
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).stroke(p.border, lineWidth: 1))
    }

    /// Every cell in a column gets that column's exact width, so a cell that wraps is measured
    /// at the width it's drawn at and its row grows to fit (no text spilling into the next row).
    private func grid(_ widths: [CGFloat]) -> some View {
        Grid(alignment: .topLeading, horizontalSpacing: 0, verticalSpacing: 0) {
            GridRow {
                ForEach(table.header.indices, id: \.self) { c in
                    cell(table.header[c], column: c, width: widths[c], header: true)
                }
            }
            Rectangle().fill(p.border).frame(height: 1).gridCellUnsizedAxes(.horizontal)
            ForEach(table.rows.indices, id: \.self) { r in
                if r > 0 {
                    Rectangle().fill(p.border.opacity(0.7)).frame(height: 1).gridCellUnsizedAxes(.horizontal)
                }
                GridRow {
                    ForEach(table.rows[r].indices, id: \.self) { c in
                        cell(table.rows[r][c], column: c, width: widths[c], header: false)
                    }
                }
            }
        }
        .fixedSize()
    }

    private func cell(_ s: String, column: Int, width: CGFloat, header: Bool) -> some View {
        let align = column < table.alignments.count ? table.alignments[column] : .leading
        let frameAlign: Alignment = align == .trailing ? .topTrailing : align == .center ? .top : .topLeading
        return Text(InlineMarkdown.attributed(s, isUser: false, palette: p, size: 10.5))
            .typo(Self.fontSize, header ? .semibold : .regular)
            .monospacedDigit()
            .lineSpacing(2)
            .foregroundStyle(header ? p.text : p.text.opacity(0.92))
            .multilineTextAlignment(align == .trailing ? .trailing : align == .center ? .center : .leading)
            .frame(width: width, alignment: frameAlign)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.horizontal, Self.padH)
            .padding(.vertical, Self.padV)
            .frame(maxHeight: .infinity, alignment: frameAlign)
            .background(header ? p.surfaceAlt : .clear)
    }

    /// Each column as wide as its widest cell's text, between 32 pt and `maxColumn`.
    private func columnWidths() -> [CGFloat] {
        table.header.indices.map { c in
            let texts = [table.header[c]] + table.rows.map { c < $0.count ? $0[c] : "" }
            let widest = texts.enumerated().map { i, t in
                TextMeasure.width(t, size: Self.fontSize, bold: i == 0)
            }.max() ?? 0
            return min(Self.maxColumn, max(32, widest))
        }
    }
}

/// Measures a cell's text the way it will be drawn (system font), ignoring markdown markers.
enum TextMeasure {
    static func width(_ markdown: String, size: CGFloat, bold: Bool) -> CGFloat {
        var plain = markdown
        for marker in ["**", "__", "`"] { plain = plain.replacingOccurrences(of: marker, with: "") }
        let codeSpans = markdown.components(separatedBy: "`").count / 2
        #if canImport(AppKit)
        let font = NSFont.systemFont(ofSize: Typo.pt(size), weight: bold ? .semibold : .regular)
        #else
        let font = UIFont.systemFont(ofSize: Typo.pt(size), weight: bold ? .semibold : .regular)
        #endif
        let w = (plain as NSString).size(withAttributes: [.font: font]).width
        // Code spans are monospaced and a touch wider; allow for them and for rounding.
        return ceil(w * (codeSpans > 0 ? 1.12 : 1.0)) + 2
    }
}

// MARK: - Charts

/// Categorical colours, in a fixed order that keeps neighbours distinguishable for
/// colour-blind readers. Validated against the bubble surfaces (#FFFFFF light, #18181E dark);
/// in light mode three slots sit under 3:1 contrast, so every chart also carries value labels
/// or its "Show data" table.
enum ChartPalette {
    static let light: [Color] = [0x2A78D6, 0xEB6834, 0x1BAF7A, 0xEDA100, 0xE87BA4, 0x008300].map { Color(hex: $0) }
    static let dark: [Color]  = [0x3987E5, 0xD95926, 0x199E70, 0xC98500, 0xD55181, 0x008300].map { Color(hex: $0) }

    static func colors(_ p: Palette) -> [Color] { p.isDark ? dark : light }
    /// Values and category names: text ink, never the series colour.
    static func secondaryInk(_ p: Palette) -> Color { p.isDark ? Color(hex: 0xC3C2B7) : Color(hex: 0x52514E) }
    static let mutedInk = Color(hex: 0x898781)
}

struct ChartBlockView: View {
    var spec: ChartSpec
    @Environment(\.palette) private var p
    @State private var showData = false
    @State private var hoverLabel: String?
    @State private var angle: Double?

    private struct Point: Identifiable {
        let id: Int
        let label: String
        let series: String
        let value: Double
    }

    /// A column chart with many or long category names can't label its axis; bars can.
    private var kind: ChartSpec.Kind {
        guard spec.kind == .column else { return spec.kind }
        let longest = spec.labels.map(\.count).max() ?? 0
        return spec.labels.count > 8 || longest > 8 ? .bar : .column
    }

    private var points: [Point] {
        var out: [Point] = []
        for s in spec.series {
            for (i, v) in s.values.enumerated() where !v.isNaN {
                out.append(Point(id: out.count, label: spec.labels[i], series: s.name, value: v))
            }
        }
        return out
    }

    private var seriesNames: [String] { spec.series.map(\.name) }
    private var colors: [Color] { ChartPalette.colors(p) }
    private var secondaryInk: Color { ChartPalette.secondaryInk(p) }
    private var muted: Color { ChartPalette.mutedInk }
    /// "Never a number on every point": label bars only while there are few enough to read.
    private var labelValues: Bool { points.count <= 24 }

    /// Bars side by side only when a category really has several series. When each category
    /// has at most one non-zero series (blind *or* through), side-by-side slots would sit empty,
    /// so the series share one bar position and colour alone tells them apart.
    private var grouped: Bool {
        guard seriesNames.count > 1 else { return false }
        return spec.labels.indices.contains { i in
            spec.series.filter { i < $0.values.count && !$0.values[i].isNaN && $0.values[i] != 0 }.count > 1
        }
    }

    /// The annotation text for a bar: nothing for a zero that only exists to keep series aligned.
    private func barLabel(_ v: Double) -> String? {
        guard labelValues, !(v == 0 && seriesNames.count > 1) else { return nil }
        return ChartSpec.format(v)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if spec.title != nil || (kind != .donut && spec.yLabel != nil) {
                VStack(alignment: .leading, spacing: 2) {
                    if let title = spec.title {
                        Text(title).typo(12.5, .semibold).foregroundStyle(p.text)
                    }
                    if kind != .donut, let measure = spec.yLabel {
                        Text(measure).typo(10.5, .medium).foregroundStyle(muted)
                    }
                }
            }
            if kind != .donut, seriesNames.count > 1 {
                LegendRow(items: Array(zip(seriesNames, colors)), ink: secondaryInk)
            }
            plot
            HStack(spacing: 8) {
                Spacer()
                Button(showData ? "Hide data" : "Show data") {
                    withAnimation(.easeOut(duration: 0.15)) { showData.toggle() }
                }
                .buttonStyle(.plain)
                .typo(10.5, .medium)
                .foregroundStyle(secondaryInk)
            }
            if showData { TableBlockView(table: spec.table) }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).stroke(p.border, lineWidth: 1))
    }

    @ViewBuilder private var plot: some View {
        switch kind {
        case .bar: barChart
        case .column: columnChart
        case .line: lineChart
        case .donut: donutChart
        }
    }

    // Horizontal bars: one row per category, name on the left, value at the bar tip. Drawn as
    // plain rows rather than with Swift Charts, whose category axis put names above hairline bars.
    private var barChart: some View {
        let maxValue = max(points.map(\.value).max() ?? 0, .leastNonzeroMagnitude)
        let longest = spec.labels.map(\.count).max() ?? 4
        let labelWidth = min(150, max(36, CGFloat(longest) * 6.5))
        return VStack(alignment: .leading, spacing: grouped ? 10 : 6) {
            ForEach(spec.labels.indices, id: \.self) { i in
                HStack(alignment: .center, spacing: 10) {
                    Text(spec.labels[i])
                        .typo(10.5, .medium)
                        .foregroundStyle(secondaryInk)
                        .lineLimit(2)
                        .multilineTextAlignment(.trailing)
                        .frame(width: labelWidth, alignment: .trailing)
                    VStack(alignment: .leading, spacing: 3) {
                        ForEach(barSeries(at: i), id: \.self) { si in
                            let v = spec.series[si].values[i]
                            HorizontalBar(fraction: v / maxValue, color: colors[si % colors.count],
                                          value: labelValues ? ChartSpec.format(v) : nil, ink: secondaryInk)
                        }
                    }
                    .overlay(alignment: .leading) { Rectangle().fill(p.border).frame(width: 1) }   // baseline
                }
            }
        }
    }

    /// Which series get a bar in row `i`: all of them when grouped; otherwise just the one with
    /// a value there (falling back to the first, so an all-zero row still shows its 0).
    private func barSeries(at i: Int) -> [Int] {
        let present = spec.series.indices.filter { !spec.series[$0].values[i].isNaN }
        if grouped { return present }
        return [present.first { spec.series[$0].values[i] != 0 } ?? present.first ?? 0]
    }

    private var columnChart: some View {
        Chart(points) { pt in
            BarMark(x: .value(spec.xLabel ?? "Item", pt.label),
                    y: .value(spec.yLabel ?? "Value", pt.value),
                    width: .fixed(22))
                .foregroundStyle(by: .value("Series", pt.series))
                .position(by: .value("Series", grouped ? pt.series : ""), axis: .horizontal,
                          span: .fixed(grouped ? CGFloat(seriesNames.count) * 25 : 22))
                .cornerRadius(4)
                .annotation(position: .top, spacing: 3) {
                    if let text = barLabel(pt.value) {
                        Text(text).typo(9.5, .medium).monospacedDigit().foregroundStyle(secondaryInk)
                    }
                }
        }
        .chartForegroundStyleScale(domain: seriesNames, range: Array(colors.prefix(seriesNames.count)))
        .chartLegend(.hidden)
        .chartYAxis { valueAxis }
        .chartXAxis {
            AxisMarks { _ in
                AxisValueLabel().font(Typo.font(10)).foregroundStyle(secondaryInk)
            }
        }
        .frame(height: 200)
    }

    // Line: a crosshair follows the pointer and lists every series' value at that x.
    private var lineChart: some View {
        Chart {
            ForEach(points) { pt in
                LineMark(x: .value(spec.xLabel ?? "x", pt.label),
                         y: .value(spec.yLabel ?? "Value", pt.value),
                         series: .value("Series", pt.series))
                    .foregroundStyle(by: .value("Series", pt.series))
                    .lineStyle(StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
                PointMark(x: .value(spec.xLabel ?? "x", pt.label),
                          y: .value(spec.yLabel ?? "Value", pt.value))
                    .foregroundStyle(by: .value("Series", pt.series))
                    .symbolSize(hoverLabel == pt.label ? 90 : 40)
            }
            if let h = hoverLabel {
                RuleMark(x: .value(spec.xLabel ?? "x", h))
                    .foregroundStyle(muted.opacity(0.5))
                    .lineStyle(StrokeStyle(lineWidth: 1))
                    .annotation(position: .top, spacing: 4,
                                overflowResolution: .init(x: .fit(to: .chart), y: .disabled)) {
                        tooltip(for: h)
                    }
            }
        }
        .chartForegroundStyleScale(domain: seriesNames, range: Array(colors.prefix(seriesNames.count)))
        .chartLegend(.hidden)
        .chartXSelection(value: $hoverLabel)
        .chartYAxis { valueAxis }
        .chartXAxis {
            AxisMarks { _ in
                AxisValueLabel().font(Typo.font(10)).foregroundStyle(secondaryInk)
            }
        }
        .frame(height: 200)
        .padding(.top, 18)      // room for the tooltip
    }

    // Donut: parts of a whole. Slices aren't labelled on the ring; the legend beside it
    // carries each value and share, and hovering a slice names it in the centre.
    private var donutChart: some View {
        let values = spec.series.first?.values ?? []
        let total = values.reduce(0, +)
        let selected = selectedSlice(values: values)
        return HStack(alignment: .center, spacing: 18) {
            Chart(points) { pt in
                SectorMark(angle: .value("Value", pt.value), innerRadius: .ratio(0.62), angularInset: 1.5)
                    .cornerRadius(3)
                    .foregroundStyle(by: .value("Item", pt.label))
                    .opacity(selected == nil || selected == pt.label ? 1 : 0.35)
            }
            .chartForegroundStyleScale(domain: spec.labels, range: Array(colors.prefix(spec.labels.count)))
            .chartLegend(.hidden)
            .chartAngleSelection(value: $angle)
            .chartBackground { proxy in
                GeometryReader { geo in
                    if let frame = proxy.plotFrame {
                        let r = geo[frame]
                        VStack(spacing: 1) {
                            if let s = selected, let i = spec.labels.firstIndex(of: s) {
                                Text(ChartSpec.format(values[i])).typo(14, .semibold).monospacedDigit().foregroundStyle(p.text)
                                Text(s).typo(9.5, .medium).foregroundStyle(secondaryInk).lineLimit(1)
                            } else {
                                Text(ChartSpec.format(total)).typo(14, .semibold).monospacedDigit().foregroundStyle(p.text)
                                Text("total").typo(9.5, .medium).foregroundStyle(muted)
                            }
                        }
                        .frame(width: r.width * 0.55)
                        .position(x: r.midX, y: r.midY)
                    }
                }
            }
            .frame(width: 150, height: 150)

            VStack(alignment: .leading, spacing: 6) {
                ForEach(Array(spec.labels.enumerated()), id: \.offset) { i, label in
                    HStack(spacing: 7) {
                        RoundedRectangle(cornerRadius: 2).fill(colors[i % colors.count]).frame(width: 10, height: 10)
                        Text(label).typo(11, .medium).foregroundStyle(p.text).lineLimit(2)
                        Spacer(minLength: 8)
                        Text(ChartSpec.format(values[i])).typo(11, .medium).monospacedDigit().foregroundStyle(secondaryInk)
                        Text(total > 0 ? "\(Int((values[i] / total * 100).rounded()))%" : "")
                            .typo(10.5, .medium).monospacedDigit().foregroundStyle(muted)
                            .frame(width: 34, alignment: .trailing)
                    }
                }
            }
            .frame(maxWidth: 250)
        }
    }

    private func selectedSlice(values: [Double]) -> String? {
        guard let angle else { return nil }
        var running = 0.0
        for (i, v) in values.enumerated() {
            running += v
            if angle <= running { return spec.labels[i] }
        }
        return nil
    }

    private func tooltip(for label: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label).typo(10.5, .semibold).foregroundStyle(p.text)
            ForEach(Array(spec.series.enumerated()), id: \.offset) { i, s in
                if let idx = spec.labels.firstIndex(of: label) {
                    HStack(spacing: 5) {
                        Circle().fill(colors[i % colors.count]).frame(width: 7, height: 7)
                        Text(s.name).typo(10, .medium).foregroundStyle(secondaryInk)
                        Spacer(minLength: 6)
                        Text(ChartSpec.format(s.values[idx])).typo(10, .semibold).monospacedDigit().foregroundStyle(p.text)
                    }
                }
            }
        }
        .padding(8)
        .frame(minWidth: 110)
        .background(p.surface, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous).stroke(p.border, lineWidth: 1))
    }

    /// The value axis: hairline gridlines, recessive labels.
    private var valueAxis: some AxisContent {
        AxisMarks { _ in
            AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5)).foregroundStyle(p.border)
            AxisValueLabel().font(Typo.font(9.5)).foregroundStyle(muted)
        }
    }
}

/// One bar: square at the baseline, 4 pt rounded at the data end, value just past the tip.
private struct HorizontalBar: View {
    var fraction: Double
    var color: Color
    var value: String?
    var ink: Color

    var body: some View {
        GeometryReader { geo in
            let room = max(0, geo.size.width - (value == nil ? 0 : 48))
            let width = fraction > 0 ? max(3, room * min(1, fraction)) : 0
            HStack(spacing: 6) {
                UnevenRoundedRectangle(topLeadingRadius: 0, bottomLeadingRadius: 0,
                                       bottomTrailingRadius: 4, topTrailingRadius: 4, style: .continuous)
                    .fill(color)
                    .frame(width: width)
                if let value {
                    Text(value).typo(10, .medium).monospacedDigit().foregroundStyle(ink).fixedSize()
                }
                Spacer(minLength: 0)
            }
        }
        .frame(height: 14)
    }
}

/// Swatch + name for each series; text stays in ink, the swatch carries the colour.
private struct LegendRow: View {
    var items: [(String, Color)]
    var ink: Color

    var body: some View {
        FlowLayout(spacing: 10, alignment: .leading) {
            ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                HStack(spacing: 5) {
                    RoundedRectangle(cornerRadius: 2).fill(item.1).frame(width: 10, height: 10)
                    Text(item.0).typo(10.5, .medium).foregroundStyle(ink)
                }
            }
        }
    }
}

// MARK: - Working

/// "4 lookups · Show working": the geometry calls and notes behind an answer, collapsed by
/// default so the answer leads.
struct WorkDisclosure: View {
    var work: [WorkStep]
    @Environment(\.palette) private var p
    @State private var open = false

    private var lookups: Int {
        work.filter { if case .lookup = $0 { return true } else { return false } }.count
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button {
                withAnimation(.easeOut(duration: 0.15)) { open.toggle() }
            } label: {
                HStack(spacing: 5) {
                    Image(systemName: "chevron.right")
                        .font(.system(size: 8, weight: .semibold))
                        .rotationEffect(.degrees(open ? 90 : 0))
                    Text("\(lookups) geometry lookup\(lookups == 1 ? "" : "s") · \(open ? "Hide" : "Show") working")
                }
                .typo(10.5, .medium)
                .foregroundStyle(p.textMuted)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            if open {
                VStack(alignment: .leading, spacing: 5) {
                    ForEach(Array(work.enumerated()), id: \.offset) { _, step in
                        switch step {
                        case .lookup(let label):
                            Label(label, systemImage: "gearshape")
                                .labelStyle(.titleAndIcon)
                                .typo(10.5, .medium)
                                .foregroundStyle(p.textMuted)
                        case .note(let note):
                            Text(note)
                                .typo(11, .regular)
                                .foregroundStyle(p.textMuted)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
                .padding(.leading, 12)
                .overlay(alignment: .leading) { Rectangle().fill(p.border).frame(width: 1) }
                .padding(.leading, 3)
            }
        }
    }
}

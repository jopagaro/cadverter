import SwiftUI

/// A wrapping horizontal layout (CSS `flex-wrap: wrap`) used for format tags and suggestion chips.
struct FlowLayout: Layout {
    var spacing: CGFloat = 5
    var alignment: HorizontalAlignment = .center

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? .infinity
        let rows = arrange(subviews: subviews, maxWidth: width)
        let height = rows.reduce(0) { $0 + $1.height } + CGFloat(max(0, rows.count - 1)) * spacing
        let maxRowWidth = rows.map(\.width).max() ?? 0
        return CGSize(width: width == .infinity ? maxRowWidth : width, height: height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let rows = arrange(subviews: subviews, maxWidth: bounds.width)
        var y = bounds.minY
        for row in rows {
            var x: CGFloat
            switch alignment {
            case .center:   x = bounds.minX + (bounds.width - row.width) / 2
            case .trailing: x = bounds.maxX - row.width
            default:        x = bounds.minX
            }
            for item in row.items {
                let size = item.size
                item.subview.place(at: CGPoint(x: x, y: y + (row.height - size.height) / 2), proposal: ProposedViewSize(size))
                x += size.width + spacing
            }
            y += row.height + spacing
        }
    }

    private struct Row { var items: [(subview: LayoutSubview, size: CGSize)] = []; var width: CGFloat = 0; var height: CGFloat = 0 }

    private func arrange(subviews: Subviews, maxWidth: CGFloat) -> [Row] {
        var rows: [Row] = []
        var current = Row()
        for sub in subviews {
            let size = sub.sizeThatFits(.unspecified)
            let extra = current.items.isEmpty ? 0 : spacing
            if current.width + extra + size.width > maxWidth, !current.items.isEmpty {
                rows.append(current)
                current = Row()
            }
            current.items.append((sub, size))
            current.width += (current.items.count == 1 ? 0 : spacing) + size.width
            current.height = max(current.height, size.height)
        }
        if !current.items.isEmpty { rows.append(current) }
        return rows
    }
}

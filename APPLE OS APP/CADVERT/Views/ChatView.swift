import SwiftUI

/// `.messages` — transcript with empty state, system pill, suggestion chips and bubbles.
struct ChatView: View {
    var isCompact: Bool
    @Environment(AppModel.self) private var model
    @Environment(\.palette) private var p

    var body: some View {
        GeometryReader { geo in
            ScrollViewReader { proxy in
                ScrollView {
                    if model.transcript.isEmpty {
                        EmptyStateView(isCompact: isCompact)
                            .frame(minHeight: geo.size.height - 32)
                    } else {
                        LazyVStack(spacing: 14) {
                            ForEach(model.transcript) { item in
                                switch item {
                                case .system(_, let text):
                                    SystemMessage(text: text)
                                case .chips(_, let questions):
                                    SuggestionChips(questions: questions)
                                case .message(let m):
                                    MessageBubble(message: m, maxWidth: geo.size.width * 0.82)
                                }
                            }
                            Color.clear.frame(height: 1).id("bottom")
                        }
                        .padding(.horizontal, isCompact ? 14 : 20)
                        .padding(.vertical, 16)
                    }
                }
                .onChange(of: model.transcript) { _, _ in
                    withAnimation(.easeOut(duration: 0.2)) { proxy.scrollTo("bottom", anchor: .bottom) }
                }
            }
        }
    }
}

/// `.empty-state`
struct EmptyStateView: View {
    var isCompact: Bool
    @Environment(AppModel.self) private var model
    @Environment(\.palette) private var p

    var body: some View {
        VStack(spacing: 10) {
            Text("🔩").font(.system(size: 40)).opacity(0.4)
            Text("Upload a CAD file to start")
                .typo(15, .semibold)
                .foregroundStyle(p.text)
                .opacity(0.45)
            Text("Then ask anything about the part — dimensions, features, manufacturability, tolerances.")
                .typo(12, .medium)
                .foregroundStyle(p.textMuted)
                .multilineTextAlignment(.center)
                .lineSpacing(3)
                .frame(maxWidth: 300)
            if isCompact {
                Button("Choose a CAD file") { model.showFileImporter = true }
                    .buttonStyle(PrimaryButtonStyle())
                    .padding(.top, 8)
            }
            Button("New here? See how it works →") { model.sheet = .about }
                .buttonStyle(PlainTextButtonStyle())
                .typo(12, .semibold)
                .foregroundStyle(p.textMuted)
                .padding(.top, 14)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(40)
    }
}

/// `.msg-system`
struct SystemMessage: View {
    var text: String
    @Environment(\.palette) private var p

    var body: some View {
        Text(attributed)
            .typo(11, .medium)
            .foregroundStyle(p.textMuted)
            .multilineTextAlignment(.center)
            .lineSpacing(3)
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .frame(maxWidth: .infinity)
            .background(p.surfaceAlt, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).stroke(p.border, lineWidth: 1))
    }

    /// Bold the filename (first " · " / " processed" segment), like `<strong>` in the web pill.
    private var attributed: AttributedString {
        var a = AttributedString(text)
        let cut = text.range(of: " processed") ?? text.range(of: " · ")
        if let cut, let end = AttributedString.Index(cut.lowerBound, within: a) {
            a[a.startIndex..<end].font = Typo.font(11, .semibold)
            a[a.startIndex..<end].foregroundColor = p.text
        }
        return a
    }
}

/// `.suggestion-chips`
struct SuggestionChips: View {
    var questions: [String]
    @Environment(AppModel.self) private var model
    @Environment(\.palette) private var p

    var body: some View {
        FlowLayout(spacing: 5, alignment: .center) {
            ForEach(questions, id: \.self) { q in
                Chip(text: q) { model.send(q) }
            }
        }
        .frame(maxWidth: .infinity)
    }
}

struct Chip: View {
    var text: String
    var action: () -> Void
    @Environment(\.palette) private var p
    @State private var hover = false

    var body: some View {
        Button(action: action) {
            Text(text)
                .typo(10.5, .medium)
                .foregroundStyle(hover ? p.accent : p.textMuted)
                .padding(.horizontal, 12)
                .padding(.vertical, 5)
                .background(p.isDark ? Color.clear : p.surface, in: Capsule())
                .overlay(Capsule().stroke(hover ? p.accent : p.border, lineWidth: 1))
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .onHover { hover = $0 }
    }
}

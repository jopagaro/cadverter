import SwiftUI

/// `.msg.user` / `.msg.assistant` — avatar + bubble, with typing dots and tool indicator states.
struct MessageBubble: View {
    var message: ChatMessage
    var maxWidth: CGFloat
    @Environment(\.palette) private var p

    private var isUser: Bool { message.role == .user }

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            if isUser { Spacer(minLength: 0) }
            if !isUser { avatar }
            bubble
                .frame(maxWidth: maxWidth, alignment: isUser ? .trailing : .leading)
            if isUser { avatar }
            if !isUser { Spacer(minLength: 0) }
        }
        .frame(maxWidth: .infinity, alignment: isUser ? .trailing : .leading)
    }

    private var avatar: some View {
        Text(isUser ? "You" : "C")
            .font(.system(size: 8 * Typo.scale, weight: .bold))
            .foregroundStyle(isUser ? p.textMuted : .white)
            .frame(width: 26, height: 26)
            .background(isUser ? p.userAvatarBg : p.accent, in: Circle())
            .overlay(Circle().stroke(isUser ? p.border : .clear, lineWidth: 1))
            .padding(.top, 2)
    }

    @ViewBuilder
    private var bubble: some View {
        let shape = UnevenRoundedRectangle(
            topLeadingRadius: 16,
            bottomLeadingRadius: isUser ? 16 : 4,
            bottomTrailingRadius: isUser ? 4 : 16,
            topTrailingRadius: 16,
            style: .continuous)

        VStack(alignment: .leading, spacing: 6) {
            if message.isTyping {
                TypingDots()
            } else {
                if let tool = message.toolLabel {
                    ToolIndicator(label: tool)
                }
                if !message.text.isEmpty {
                    MarkdownText(text: message.text, isUser: isUser)
                }
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(isUser ? p.accent : p.surface, in: shape)
        .overlay(shape.stroke(isUser ? .clear : (message.isError ? p.danger.opacity(0.4) : p.border), lineWidth: 1))
        .textSelection(.enabled)
    }
}

/// Renders the assistant's markdown-ish text: paragraphs (inline **bold**, *em*, `code`), lists, code blocks.
struct MarkdownText: View {
    var text: String
    var isUser: Bool
    @Environment(\.palette) private var p

    var body: some View {
        let blocks = MarkdownBlocks.parse(text)
        VStack(alignment: .leading, spacing: 6) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                switch block {
                case .paragraph(let s):
                    Text(inline(s))
                        .typo(12.5, .medium)
                        .lineSpacing(3)
                        .foregroundStyle(isUser ? .white : p.text)
                        .fixedSize(horizontal: false, vertical: true)
                case .list(let items):
                    VStack(alignment: .leading, spacing: 3) {
                        ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                            HStack(alignment: .top, spacing: 6) {
                                Text("•").typo(12.5, .medium)
                                Text(inline(item)).typo(12.5, .medium).lineSpacing(3)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                            .foregroundStyle(isUser ? .white : p.text)
                        }
                    }
                    .padding(.leading, 8)
                case .code(let code):
                    ScrollView(.horizontal, showsIndicators: false) {
                        Text(code)
                            .typo(11, .regular, mono: true)
                            .foregroundStyle(isUser ? .white : p.text)
                            .padding(12)
                    }
                    .background(isUser ? Color.white.opacity(0.15) : p.bg, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
                }
            }
        }
    }

    private func inline(_ s: String) -> AttributedString {
        var a = (try? AttributedString(markdown: s, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)))
            ?? AttributedString(s)
        // Inline code → mono with a soft background, like `.msg-bubble code`.
        for run in a.runs where run.inlinePresentationIntent?.contains(.code) == true {
            a[run.range].font = Typo.font(10.5, .medium, mono: true)
            a[run.range].backgroundColor = isUser ? Color.white.opacity(0.2) : p.surfaceAlt
        }
        return a
    }
}

/// `.typing-dot` ×3
struct TypingDots: View {
    @Environment(\.palette) private var p
    @State private var on = false

    var body: some View {
        HStack(spacing: 4) {
            ForEach(0..<3, id: \.self) { i in
                Circle()
                    .fill(p.textMuted)
                    .frame(width: 6, height: 6)
                    .opacity(on ? 1 : 0.2)
                    .animation(.easeInOut(duration: 0.6).repeatForever(autoreverses: true).delay(Double(i) * 0.2), value: on)
            }
        }
        .padding(.vertical, 4)
        .onAppear { on = true }
    }
}

/// `.tool-indicator` — "⚙ Computing exact distance…" pill with a pulse.
struct ToolIndicator: View {
    var label: String
    @Environment(\.palette) private var p
    @State private var pulse = false

    var body: some View {
        HStack(spacing: 5) {
            Image(systemName: "gearshape.fill").font(.system(size: 9))
            Text("\(label)…")
        }
        .typo(10, .medium)
        .foregroundStyle(p.isDark ? p.accent : p.textMuted)
        .padding(.horizontal, 10)
        .padding(.vertical, 3)
        .background(p.isDark ? p.accent.opacity(0.08) : p.surfaceAlt, in: Capsule())
        .overlay(Capsule().stroke(p.isDark ? p.accent.opacity(0.15) : p.border, lineWidth: 1))
        .opacity(pulse ? 0.45 : 1)
        .animation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true), value: pulse)
        .onAppear { pulse = true }
    }
}

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
                if !message.work.isEmpty {
                    WorkDisclosure(work: message.work)
                }
                if let tool = message.toolLabel {
                    ToolIndicator(label: tool)
                }
                if !message.text.isEmpty {
                    MarkdownText(text: message.text, isUser: isUser)
                }
                if let shown = message.highlight {
                    ShowInViewButton(highlight: shown)
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

/// Under an answer that pointed at something: shows it in the 3D view again.
struct ShowInViewButton: View {
    var highlight: ViewerHighlight
    @Environment(AppModel.self) private var model
    @Environment(\.palette) private var p
    @State private var hover = false

    private var isShown: Bool { model.answerHighlight == highlight }

    var body: some View {
        Button { model.show(highlight) } label: {
            HStack(spacing: 5) {
                Image(systemName: "scope").font(.system(size: 9.5, weight: .semibold))
                Text(isShown ? "Shown in 3D view" : "Show in 3D view")
                if !highlight.label.isEmpty {
                    Text("· \(highlight.label)").foregroundStyle(p.textDim).lineLimit(1)
                }
            }
            .typo(10, .medium)
            .foregroundStyle(isShown || hover ? p.accent : p.textMuted)
            .padding(.horizontal, 9)
            .padding(.vertical, 3)
            .background(isShown ? p.accentSoft : (hover ? p.surfaceAlt : .clear), in: Capsule())
            .overlay(Capsule().stroke(isShown ? p.accent.opacity(0.35) : p.border, lineWidth: 1))
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .onHover { hover = $0 }
        .help("Highlight this in the 3D view and turn to it")
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

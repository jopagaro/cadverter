import SwiftUI

// MARK: - Logo

/// The `.logo-mark` — dark rounded square with a bold "C".
struct LogoMark: View {
    var size: CGFloat = 24
    @Environment(\.palette) private var p

    var body: some View {
        Text("C")
            .font(.system(size: size * 0.5, weight: .heavy))
            .foregroundStyle(.white)
            .frame(width: size, height: size)
            .background(p.logoBg, in: RoundedRectangle(cornerRadius: size * 0.29, style: .continuous))
    }
}

struct LogoWrap: View {
    var markSize: CGFloat = 24
    var textSize: CGFloat = 15
    @Environment(\.palette) private var p

    var body: some View {
        HStack(spacing: 8) {
            LogoMark(size: markSize)
            Text("CADVERT")
                .typo(textSize, .bold)
                .tracking(-0.3)
                .foregroundStyle(p.text)
        }
    }
}

// MARK: - Labels & badges

/// `.section-label` — 9px, 700, uppercase, letter-spaced, muted.
struct SectionLabel: View {
    var text: String
    @Environment(\.palette) private var p
    var body: some View {
        Text(text.uppercased())
            .typo(9, .bold)
            .tracking(0.8)
            .foregroundStyle(p.textMuted)
    }
}

enum BadgeStyle { case blue, green, orange }

struct Badge: View {
    var text: String
    var style: BadgeStyle = .blue
    @Environment(\.palette) private var p

    var body: some View {
        Text(text)
            .typo(9, .semibold)
            .padding(.horizontal, 7)
            .padding(.vertical, 1.5)
            .foregroundStyle(fg)
            .background(bg, in: RoundedRectangle(cornerRadius: 4, style: .continuous))
    }

    private var fg: Color {
        switch style { case .blue: return .white; case .green: return p.success; case .orange: return p.warning }
    }
    private var bg: Color {
        switch style { case .blue: return p.accent; case .green: return p.successBg; case .orange: return p.warningBg }
    }
}

/// `.fmt-tag` — small mono extension tag; `full` variant is green.
struct FormatTag: View {
    var ext: String
    var full: Bool
    @Environment(\.palette) private var p
    var body: some View {
        Text(".\(ext)")
            .typo(9, .medium, mono: true)
            .padding(.horizontal, 5)
            .padding(.vertical, 2)
            .foregroundStyle(full ? p.success : p.textMuted)
            .background(full ? p.successBg : p.surfaceAlt, in: RoundedRectangle(cornerRadius: 4, style: .continuous))
    }
}

// MARK: - Buttons

/// `.btn-primary`
struct PrimaryButtonStyle: ButtonStyle {
    @Environment(\.palette) private var p
    @Environment(\.isEnabled) private var enabled
    var compact = false

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .typo(compact ? 11 : 12, .medium)
            .foregroundStyle(.white)
            .padding(.horizontal, compact ? 12 : 16)
            .padding(.vertical, compact ? 5 : 8)
            .background(p.accent, in: RoundedRectangle(cornerRadius: compact ? 8 : 10, style: .continuous))
            .opacity(!enabled ? 0.4 : (configuration.isPressed ? 0.76 : 1))
            .animation(.easeOut(duration: 0.15), value: configuration.isPressed)
            .contentShape(Rectangle())
    }
}

/// `.btn.btn-ghost` — bordered, muted text.
struct GhostButtonStyle: ButtonStyle {
    @Environment(\.palette) private var p
    @Environment(\.isEnabled) private var enabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .typo(11, .medium)
            .foregroundStyle(configuration.isPressed ? p.text : p.textMuted)
            .padding(.horizontal, 12)
            .padding(.vertical, 5)
            .background(configuration.isPressed ? p.accentSoft : .clear, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).stroke(p.border, lineWidth: 1))
            .opacity(enabled ? 1 : 0.5)
            .contentShape(Rectangle())
    }
}

/// Header nav links ("About", "For Developers") — 12px 600, bordered pill.
struct NavLinkStyle: ButtonStyle {
    @Environment(\.palette) private var p
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .typo(12, .semibold)
            .foregroundStyle(configuration.isPressed ? p.text : p.textMuted)
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(configuration.isPressed ? p.surfaceAlt : .clear, in: RoundedRectangle(cornerRadius: 7, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 7, style: .continuous).stroke(p.border, lineWidth: 1))
            .contentShape(Rectangle())
    }
}

/// Plain text-only button (links inside copy, "View HSD →", chips…).
struct PlainTextButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .opacity(configuration.isPressed ? 0.6 : 1)
            .contentShape(Rectangle())
    }
}

// MARK: - Theme toggle

/// `.theme-toggle` with Light / Dark pills (+ System).
struct ThemeToggle: View {
    @Binding var theme: ThemePreference
    var showSystem = true
    @Environment(\.palette) private var p

    var body: some View {
        HStack(spacing: 1) {
            ForEach(ThemePreference.allCases.filter { showSystem || $0 != .system }) { t in
                Button { theme = t } label: {
                    Text(t.label)
                        .typo(11, .medium)
                        .foregroundStyle(theme == t ? p.text : p.textMuted)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 3)
                        .background(
                            RoundedRectangle(cornerRadius: 6, style: .continuous)
                                .fill(theme == t ? p.surface : .clear)
                                .shadow(color: theme == t ? .black.opacity(0.08) : .clear, radius: 1.5, y: 1)
                        )
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
        .padding(2)
        .background(p.surfaceAlt, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).stroke(p.border, lineWidth: 1))
    }
}

// MARK: - Text input chrome

/// `input[type=text]` look — surface-alt, bordered, radius 8, mono.
struct FieldChrome: ViewModifier {
    @Environment(\.palette) private var p
    var mono = true
    var radius: CGFloat = 8
    func body(content: Content) -> some View {
        content
            .textFieldStyle(.plain)
            .typo(11, .medium, mono: mono)
            .foregroundStyle(p.text)
            .padding(.horizontal, 9)
            .padding(.vertical, 6)
            .background(p.surfaceAlt, in: RoundedRectangle(cornerRadius: radius, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: radius, style: .continuous).stroke(p.border, lineWidth: 1))
    }
}

extension View {
    func fieldChrome(mono: Bool = true, radius: CGFloat = 8) -> some View {
        modifier(FieldChrome(mono: mono, radius: radius))
    }
}

// MARK: - Misc

struct Hairline: View {
    @Environment(\.palette) private var p
    var vertical = false
    var body: some View {
        Rectangle().fill(p.border)
            .frame(width: vertical ? 1 : nil, height: vertical ? nil : 1)
    }
}

/// The processing spinner: 2.5px ring with an accent-coloured cap, spinning.
struct SpinnerRing: View {
    var size: CGFloat = 44
    @Environment(\.palette) private var p
    @State private var spin = false
    var body: some View {
        ZStack {
            Circle().stroke(p.surfaceAlt, lineWidth: 2.5)
            Circle().trim(from: 0, to: 0.25).stroke(p.accent, style: StrokeStyle(lineWidth: 2.5, lineCap: .butt))
                .rotationEffect(.degrees(spin ? 360 : 0))
                .animation(.linear(duration: 0.8).repeatForever(autoreverses: false), value: spin)
        }
        .frame(width: size, height: size)
        .onAppear { spin = true }
    }
}

/// Closes a modal — the `.modal-close` circle.
struct CloseCircle: View {
    var action: () -> Void
    @Environment(\.palette) private var p
    var body: some View {
        Button(action: action) {
            Text("×")
                .font(.system(size: 14, weight: .regular))
                .foregroundStyle(p.textMuted)
                .frame(width: 24, height: 24)
                .background(p.surfaceAlt, in: Circle())
        }
        .buttonStyle(.plain)
        .keyboardShortcut(.cancelAction)
        .accessibilityLabel("Close")
    }
}

import SwiftUI

/// The 48px `<header>`: logo, About / For Developers, theme pills, status + settings.
struct HeaderBar: View {
    var isCompact: Bool
    @Environment(AppModel.self) private var model
    @Environment(\.palette) private var p

    var body: some View {
        @Bindable var settings = model.settings
        HStack(spacing: 12) {
            LogoWrap()

            if !isCompact {
                Button("About") { model.sheet = .about }
                    .buttonStyle(NavLinkStyle())
                Button("For Developers") { model.sheet = .developers }
                    .buttonStyle(NavLinkStyle())
                ThemeToggle(theme: $settings.theme, showSystem: false)
            }

            Spacer(minLength: 8)

            if let counter = model.freeCounter, !isCompact {
                FreeCounterPill(counter: counter)
            }

            if !isCompact { ProviderPill() }
            EngineStatusPill()

            if isCompact {
                Button { model.showPartSheet = true } label: {
                    Image(systemName: "cube")
                        .font(.system(size: 15, weight: .medium))
                        .foregroundStyle(p.textMuted)
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Part")
            }

            Button { model.sheet = .settings } label: {
                Image(systemName: "gearshape")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(p.textMuted)
                    .frame(width: 28, height: 28)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Settings")
        }
        .padding(.horizontal, isCompact ? 14 : 20)
        .frame(height: 48)
        .background(p.surface)
    }
}

/// `.free-counter`
struct FreeCounterPill: View {
    var counter: FreeCounter
    @Environment(\.palette) private var p

    var body: some View {
        Text(counter.text)
            .typo(10, .medium)
            .foregroundStyle(color)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(p.surfaceAlt, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous).stroke(borderColor, lineWidth: 1))
            .lineLimit(1)
    }

    private var color: Color {
        switch counter.level { case .normal: return p.textMuted; case .warn: return p.warning; case .gone: return p.danger }
    }
    private var borderColor: Color {
        switch counter.level { case .normal: return p.border; case .warn: return p.warningBg; case .gone: return p.danger.opacity(0.2) }
    }
}

/// Engine / server connection state; tap to open Settings.
struct EngineStatusPill: View {
    @Environment(AppModel.self) private var model
    @Environment(\.palette) private var p

    var body: some View {
        let status = model.engineStatus
        Button { model.sheet = .settings } label: {
            HStack(spacing: 6) {
                Circle().fill(dot(status.tone)).frame(width: 6, height: 6)
                Text(status.text)
                    .typo(10, .medium)
                    .foregroundStyle(p.textMuted)
                    .lineLimit(1)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(p.surfaceAlt, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous).stroke(p.border, lineWidth: 1))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help("Engine status — click for Settings")
    }

    private func dot(_ tone: AppModel.StatusTone) -> Color {
        switch tone {
        case .ok: return p.success
        case .busy: return p.warning
        case .bad: return p.danger
        case .off: return p.textDim
        }
    }
}

/// Which AI is answering — tap to change in Settings.
struct ProviderPill: View {
    @Environment(AppModel.self) private var model
    @Environment(\.palette) private var p

    var body: some View {
        let pr = model.settings.provider
        let label: String = {
            switch pr {
            case .apple: return "Apple Intelligence"
            case .openai, .anthropic:
                return ChatModel.models(for: pr).first { $0.id == model.settings.model(for: pr) }?.label ?? pr.label
            }
        }()
        Button { model.sheet = .settings } label: {
            HStack(spacing: 5) {
                Image(systemName: pr == .apple ? "apple.intelligence" : "sparkles")
                    .font(.system(size: 9, weight: .semibold))
                Text(label).typo(10, .medium).lineLimit(1)
            }
            .foregroundStyle(p.textMuted)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(p.surfaceAlt, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous).stroke(p.border, lineWidth: 1))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help("AI provider — click to change")
    }
}

import SwiftUI

/// Shared chrome for the `.wall-box` modals.
struct WallBox<Content: View>: View {
    var icon: String
    var title: String
    var subtitle: String
    @ViewBuilder var content: Content
    @Environment(AppModel.self) private var model
    @Environment(\.palette) private var p

    var body: some View {
        VStack(spacing: 0) {
            HStack { Spacer(); CloseCircle { model.sheet = nil } }
                .padding(.trailing, 12).padding(.top, 12)
            VStack(spacing: 0) {
                Text(icon).font(.system(size: 28)).opacity(0.7).padding(.bottom, 14)
                Text(title).typo(15, .bold).foregroundStyle(p.text).multilineTextAlignment(.center).padding(.bottom, 8)
                Text(subtitle).typo(12, .medium).foregroundStyle(p.textMuted).multilineTextAlignment(.center)
                    .lineSpacing(3).padding(.bottom, 20)
                content
            }
            .padding(.horizontal, 36)
            .padding(.bottom, 32)
        }
        .frame(maxWidth: 440)
        .background(p.surface)
        #if os(macOS)
        .frame(width: 440)
        #endif
    }
}

/// `.wall-input-row` — key field + button.
struct KeyInputRow: View {
    @Binding var key: String
    var placeholder: String = "sk-..."
    var buttonTitle: String
    var action: () -> Void
    @Environment(\.palette) private var p

    var body: some View {
        HStack(spacing: 8) {
            SecureField(placeholder, text: $key)
                .fieldChrome()
                .onSubmit(action)
                #if os(iOS)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                #endif
            Button(buttonTitle, action: action).buttonStyle(PrimaryButtonStyle(compact: true))
        }
    }
}

/// Local Mac engine: chat through a hosted provider needs the user's own key (also the first-run welcome).
struct KeyNeededView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.palette) private var p
    @State private var key = ""
    @State private var provider: AIProvider = .openai

    var body: some View {
        WallBox(icon: "🔑",
                title: "Add your \(provider.keyVendor) key to chat",
                subtitle: "Geometry analysis runs entirely on this Mac. To ask questions through \(provider.label), the app calls \(provider.keyVendor) with your own API key — stored in your Keychain and sent only to \(provider.keyVendor).") {
            if AIProvider.selectable.count > 1 {
                Picker("Provider", selection: $provider) {
                    ForEach(AIProvider.selectable.filter(\.needsKey)) { pr in Text(pr.label).tag(pr) }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .padding(.bottom, 12)
                .onChange(of: provider) { _, new in key = model.settings.key(for: new) }
            }
            KeyInputRow(key: $key, placeholder: provider.keyPlaceholder, buttonTitle: "Continue") {
                Task { await model.saveAPIKey(key, for: provider) }
            }
            Text("\(provider.keyHint) You can change it later in Settings.")
                .typo(11, .medium).foregroundStyle(p.textDim).multilineTextAlignment(.center).padding(.top, 12)
            if AppleIntelligence.isSupportedOS {
                Button("Use Apple Intelligence instead — free, on-device, no key") {
                    model.settings.provider = .apple
                    model.sheet = nil
                    if let pending = model.pendingMessage { model.pendingMessage = nil; model.send(pending) }
                }
                .buttonStyle(PlainTextButtonStyle())
                .typo(11, .medium).foregroundStyle(p.accent)
                .padding(.top, 10)
            }
            Button("Not now — just analyse files") { model.sheet = nil }
                .buttonStyle(PlainTextButtonStyle())
                .typo(11, .medium).foregroundStyle(p.textDim).underline()
                .padding(.top, 8)
        }
        .onAppear {
            provider = model.settings.provider.needsKey ? model.settings.provider : .openai
            key = model.settings.key(for: provider)
        }
    }
}

/// `#byokModal` — free messages used up on a hosted server.
struct ByokWallView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.palette) private var p
    @State private var key = ""

    private var provider: AIProvider { model.settings.provider.needsKey ? model.settings.provider : .openai }

    var body: some View {
        WallBox(icon: "🔑",
                title: "You've used your 3 free messages",
                subtitle: "Keep going with your own \(provider.keyVendor) key — it is sent straight to \(provider.keyVendor) and never stored on the server. Plans are managed on the CADVERT website.") {
            KeyInputRow(key: $key, placeholder: provider.keyPlaceholder, buttonTitle: "Continue") {
                Task { await model.saveAPIKey(key, for: provider) }
            }
            Text("Come back tomorrow for 3 more free messages.")
                .typo(11, .medium).foregroundStyle(p.textDim).padding(.top, 12)
        }
    }
}

/// `#byokKeyModal` — paid BYOK user without a key entered.
struct ByokKeyOnlyView: View {
    @Environment(AppModel.self) private var model
    @State private var key = ""

    private var provider: AIProvider { model.settings.provider.needsKey ? model.settings.provider : .openai }

    var body: some View {
        WallBox(icon: "🔑",
                title: "Enter your \(provider.keyVendor) key",
                subtitle: "You're on the BYOK plan — paste your key below. It's sent directly to \(provider.keyVendor) and never stored on the server.") {
            KeyInputRow(key: $key, placeholder: provider.keyPlaceholder, buttonTitle: "Continue") {
                Task { await model.saveAPIKey(key, for: provider) }
            }
        }
    }
}

/// `#fileLimitModal`
struct FileLimitView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.palette) private var p

    var body: some View {
        WallBox(icon: "📁",
                title: "Daily file limit reached",
                subtitle: "You've used your free file for today on this server. Limits reset at midnight.\nThe Mac app analyses unlimited files, fully offline.") {
            Button("Dismiss — come back tomorrow") { model.sheet = nil }
                .buttonStyle(PlainTextButtonStyle())
                .typo(11, .medium).foregroundStyle(p.textDim).underline()
        }
    }
}

/// Remote server requires Google sign-in, which the native app does not do yet.
struct SignInNeededView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.palette) private var p

    var body: some View {
        WallBox(icon: "🔒",
                title: "This server requires sign-in",
                subtitle: "The hosted CADVERT server checks a Google account token on every upload. Run your own server with DISABLE_AUTH=1, or paste an access token under Settings → Advanced.") {
            Button("Open Settings") { model.sheet = .settings }.buttonStyle(PrimaryButtonStyle())
        }
    }
}

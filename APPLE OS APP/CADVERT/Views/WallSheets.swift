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

/// Shown when someone picks Apple Intelligence on a Mac that cannot run it.
///
/// The three reasons need different answers — wrong hardware is permanent, switched off
/// and still-downloading are not — so the sheet carries the specific guidance rather than
/// a single generic line, and always offers the way forward that always works: your own key.
struct AppleUnavailableView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.palette) private var p
    @State private var key = ""
    @State private var provider: AIProvider = .openai

    var body: some View {
        let status = AppleIntelligence.status
        WallBox(icon: "✦",
                title: status.detail,
                subtitle: status.guidance ?? "Add your own API key to ask questions about this part.") {
            Picker("Provider", selection: $provider) {
                ForEach(AIProvider.selectable.filter(\.needsKey)) { pr in Text(pr.label).tag(pr) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .padding(.bottom, 10)
            .onChange(of: provider) { _, new in key = model.settings.key(for: new) }

            KeyInputRow(key: $key, placeholder: provider.keyPlaceholder, buttonTitle: "Save & continue") {
                Task { await model.saveAPIKey(key, for: provider) }
            }
            Text(provider.keyHint)
                .typo(11, .medium).foregroundStyle(p.textDim)
                .multilineTextAlignment(.center).padding(.top, 10)

            if AppleIntelligence.isDeviceCapable {
                Button("Try Apple Intelligence again") {
                    if AppleIntelligence.status.available {
                        model.sheet = nil
                        if let pending = model.pendingMessage { model.pendingMessage = nil; model.send(pending) }
                    } else {
                        model.showToast(AppleIntelligence.status.detail)
                    }
                }
                .buttonStyle(PlainTextButtonStyle())
                .typo(11, .medium).foregroundStyle(p.accent)
                .padding(.top, 12)
            }

            Button("Not now — analysis works without AI") { model.sheet = nil }
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

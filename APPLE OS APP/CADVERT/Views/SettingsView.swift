import SwiftUI

/// Engine, AI key/model, appearance. `embedded` = macOS Settings window (no close button).
struct SettingsView: View {
    var embedded: Bool
    @Environment(AppModel.self) private var model
    @Environment(\.palette) private var p
    @Environment(\.colorScheme) private var scheme
    @State private var keyDraft = ""
    @State private var keySaved = false

    var body: some View {
        @Bindable var settings = model.settings
        let pal = Palette.forScheme(scheme)
        VStack(spacing: 0) {
            if !embedded {
                HStack {
                    Text("Settings").typo(13, .semibold).foregroundStyle(pal.text)
                    Spacer()
                    CloseCircle { model.sheet = nil }
                }
                .padding(.horizontal, 20).padding(.vertical, 14)
                Hairline()
            }
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    // ── Engine ──
                    VStack(alignment: .leading, spacing: 10) {
                        SectionLabel(text: "Engine")
                        if EngineMode.available.count > 1 {
                            Picker("Engine", selection: $settings.engineMode) {
                                Text("On this Mac (offline)").tag(EngineMode.local)
                                Text("Remote server").tag(EngineMode.remote)
                            }
                            .pickerStyle(.segmented)
                            .labelsHidden()
                            .onChange(of: settings.engineMode) { _, _ in Task { await model.connect() } }
                        }

                        HStack(spacing: 6) {
                            Circle().fill(statusColor(pal)).frame(width: 6, height: 6)
                            Text(model.engineStatus.text).typo(11, .medium).foregroundStyle(pal.textMuted)
                            Spacer()
                            Button(settings.engineMode == .local ? "Restart" : "Reconnect") { Task { await model.connect() } }
                                .buttonStyle(GhostButtonStyle())
                        }
                        if let f = model.connection.failureMessage {
                            Text(f).typo(10.5, .medium).foregroundStyle(pal.danger).lineSpacing(2)
                                .fixedSize(horizontal: false, vertical: true)
                        }

                        if settings.engineMode == .remote {
                            Text("Address of a cadvert-server (the same server that powers the web app). For iPhone/iPad this is the only mode.")
                                .typo(10.5, .medium).foregroundStyle(pal.textDim).fixedSize(horizontal: false, vertical: true)
                            HStack(spacing: 8) {
                                TextField("https://cadvert.example.com", text: $settings.remoteURLString)
                                    .fieldChrome()
                                    .onSubmit { Task { await model.connect() } }
                                    #if os(iOS)
                                    .keyboardType(.URL).textInputAutocapitalization(.never).autocorrectionDisabled()
                                    #endif
                                Button("Connect") { Task { await model.connect() } }.buttonStyle(PrimaryButtonStyle(compact: true))
                            }
                        } else {
                            #if os(macOS)
                            Text(EngineLocator.bundledPython() != nil
                                 ? "A CADVERT engine is bundled inside this app — nothing to install."
                                 : "No engine is bundled in this build. The app also looks in ~/.cadvert/engine and at the path below (Debug builds only, since the App Sandbox blocks executables outside the app).")
                                .typo(10.5, .medium).foregroundStyle(pal.textDim).fixedSize(horizontal: false, vertical: true)
                            HStack(spacing: 8) {
                                TextField("Custom Python (e.g. ~/cadvert/.venv/bin/python)", text: $settings.pythonPath)
                                    .fieldChrome()
                                    .onSubmit { Task { await model.connect() } }
                                Button("Use") { Task { await model.connect() } }.buttonStyle(GhostButtonStyle())
                            }
                            Button("Engine log…") { model.refreshEngineLog(); model.sheet = .engineLog }
                                .buttonStyle(GhostButtonStyle())
                            #endif
                        }
                    }

                    // ── AI ──
                    VStack(alignment: .leading, spacing: 10) {
                        SectionLabel(text: "AI")
                        Picker("Provider", selection: $settings.provider) {
                            ForEach(AIProvider.selectable) { pr in Text(pr.label).tag(pr) }
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()
                        .onChange(of: settings.provider) { _, _ in keyDraft = settings.key(for: settings.provider); keySaved = !keyDraft.isEmpty }

                        if settings.provider == .apple {
                            let st = AppleIntelligence.status
                            HStack(alignment: .top, spacing: 6) {
                                Circle().fill(st.available ? pal.success : pal.warning).frame(width: 6, height: 6).padding(.top, 5)
                                Text(st.detail).typo(11, .medium).foregroundStyle(pal.text).fixedSize(horizontal: false, vertical: true)
                            }
                            if let guidance = st.guidance {
                                Text(guidance)
                                    .typo(10.5, .medium).foregroundStyle(pal.textMuted)
                                    .lineSpacing(2).fixedSize(horizontal: false, vertical: true)
                                    .padding(.leading, 12)
                            }
                            Text("Apple's on-device model is quick and private but small (about 4k tokens of context). Great for lookups; switch to OpenAI or Claude for big assemblies or deeper engineering judgement.")
                                .typo(10.5, .medium).foregroundStyle(pal.textDim).fixedSize(horizontal: false, vertical: true)
                        } else {
                            let pr = settings.provider
                            Text(settings.engineMode == .local
                                 ? "The offline engine calls \(pr.keyVendor) with your own key (kept in your Keychain). Required for chat; not needed to analyse files."
                                 : "Optional: your own \(pr.keyVendor) key (BYOK). Sent as a request header straight to the server, never stored there.")
                                .typo(10.5, .medium).foregroundStyle(pal.textDim).fixedSize(horizontal: false, vertical: true)
                            HStack(spacing: 8) {
                                SecureField(pr.keyPlaceholder, text: $keyDraft)
                                    .fieldChrome()
                                    #if os(iOS)
                                    .textInputAutocapitalization(.never).autocorrectionDisabled()
                                    #endif
                                Button(keySaved ? "Saved" : "Save") {
                                    Task { await model.saveAPIKey(keyDraft, for: pr); keySaved = true }
                                }
                                .buttonStyle(PrimaryButtonStyle(compact: true))
                                .disabled(keyDraft.trimmingCharacters(in: .whitespaces).isEmpty)
                                if !settings.key(for: pr).isEmpty {
                                    Button("Remove") {
                                        settings.setKey("", for: pr); keyDraft = ""; keySaved = false
                                        if settings.engineMode == .local { Task { await model.connect() } }
                                    }
                                    .buttonStyle(GhostButtonStyle())
                                }
                            }
                            Text(pr.keyHint).typo(10.5, .medium).foregroundStyle(pal.textDim)
                            HStack(spacing: 8) {
                                Text("Model").typo(11, .medium).foregroundStyle(pal.textMuted)
                                Picker("Model", selection: Binding(get: { settings.model(for: pr) }, set: { settings.setModel($0, for: pr) })) {
                                    ForEach(ChatModel.models(for: pr), id: \.id) { m in Text(m.label).tag(m.id) }
                                }
                                .labelsHidden()
                                .frame(maxWidth: 220)
                            }
                            if let info = model.serverConfig?.info(for: pr), info.available == false {
                                Text("This server doesn't have the \(pr.keyVendor) SDK installed (pip install cadvert[llm]).")
                                    .typo(10.5, .medium).foregroundStyle(pal.danger)
                            }
                        }
                    }

                    // ── Storage ──
                    VStack(alignment: .leading, spacing: 10) {
                        SectionLabel(text: "Storage")
                        Text("Opened parts are cached so views and the geometry document reopen "
                             + "instantly. All of it rebuilds from your original CAD file, so "
                             + "clearing it loses nothing.")
                            .typo(10.5, .medium).foregroundStyle(pal.textDim)
                            .fixedSize(horizontal: false, vertical: true)
                        HStack(spacing: 8) {
                            Text(model.cacheUsage?.summary ?? "Checking…")
                                .typo(11, .medium).foregroundStyle(pal.text)
                            Spacer()
                            Button(model.clearingCache ? "Clearing…" : "Clear cache") {
                                Task { await model.clearCache() }
                            }
                            .buttonStyle(GhostButtonStyle())
                            .disabled(model.clearingCache || (model.cacheUsage?.sessions ?? 0) == 0)
                        }
                        if let hours = model.cacheUsage?.ttlHours {
                            Text("Cached parts are removed automatically after \(hours) hours.")
                                .typo(10.5, .medium).foregroundStyle(pal.textDim)
                        }
                    }

                    // ── Appearance ──
                    VStack(alignment: .leading, spacing: 10) {
                        SectionLabel(text: "Appearance")
                        ThemeToggle(theme: $settings.theme, showSystem: true)
                    }

                    // ── About ──
                    VStack(alignment: .leading, spacing: 6) {
                        SectionLabel(text: "About")
                        Text("CADVERT \(model.appVersion) — exact CAD geometry for AI.")
                            .typo(11, .medium).foregroundStyle(pal.textMuted)
                        HStack(spacing: 12) {
                            Button("How it works") { model.sheet = .about }.buttonStyle(NavLinkStyle())
                            Button("For Developers") { model.sheet = .developers }.buttonStyle(NavLinkStyle())
                            Link("PyPI", destination: URL(string: "https://pypi.org/project/cadvert/")!)
                                .typo(11, .semibold).foregroundStyle(pal.textMuted)
                        }
                    }
                }
                .padding(24)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .background(pal.surface)
        .environment(\.palette, pal)
        .onAppear { keyDraft = model.settings.key(for: model.settings.provider); keySaved = !keyDraft.isEmpty }
        .task { await model.refreshCacheUsage() }
        #if os(macOS)
        .frame(width: 560, height: embedded ? nil : 620)
        #endif
    }

    private func statusColor(_ pal: Palette) -> Color {
        switch model.engineStatus.tone {
        case .ok: return pal.success
        case .busy: return pal.warning
        case .bad: return pal.danger
        case .off: return pal.textDim
        }
    }
}

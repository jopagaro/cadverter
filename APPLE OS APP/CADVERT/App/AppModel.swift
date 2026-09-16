import Foundation
import SwiftUI
import Observation
import UniformTypeIdentifiers

enum ConnectionState: Equatable {
    case idle
    case connecting(String)
    case ready
    case failed(String)

    var isReady: Bool { if case .ready = self { return true } else { return false } }
    var failureMessage: String? { if case .failed(let m) = self { return m } else { return nil } }
}

struct Toast: Equatable {
    var text: String
    var success: Bool = false
}

enum AppSheet: String, Identifiable {
    case hsd, about, developers, settings, keyNeeded, engineLog
    var id: String { rawValue }
}

/// Central application state. Mirrors the globals + handlers in `index.html`
/// (sessionId, hsdText, chatHistory, streaming, messagesSent, byokKey …) as one observable object.
@MainActor
@Observable
final class AppModel {
    static let shared = AppModel()

    let settings: AppSettings

    // Connection / engine
    var connection: ConnectionState = .idle
    private(set) var client: CadvertClient?
    var serverConfig: ServerConfig?
    var engineLog: String = ""
    var engineTrail: String = ""

    // Part session + transcript
    var session: PartSession?
    var transcript: [ChatItem] = []
    var processing: ProcessingState?
    var streaming = false
    var messagesSent = 0
    var doneLabel: String?
    var draft = ""
    var pendingMessage: String?
    private var lastImport: (data: Data, filename: String)?

    // UI
    var toast: Toast?
    var sheet: AppSheet?
    var lightboxURL: URL?
    var showFileImporter = false
    var showHSDExporter = false
    var showPartSheet = false
    var isDropTargeted = false

    #if os(macOS)
    private let engine = LocalEngine()
    #endif
    private let appleChat = AppleIntelligence()
    private var connectTask: Task<Void, Never>?
    private var toastTask: Task<Void, Never>?
    private var didAutoConnect = false

    init(settings: AppSettings = AppSettings()) {
        self.settings = settings
    }

    // MARK: - Derived state

    var canChat: Bool { session != nil && !streaming && connection.isReady }

    enum StatusTone { case ok, busy, bad, off }

    var engineStatus: (text: String, tone: StatusTone) {
        switch (settings.engineMode, connection) {
        case (.local, .idle):              return ("Engine off", .off)
        case (.local, .connecting(let s)): return (s, .busy)
        case (.local, .ready):             return ("Local engine · ready", .ok)
        case (.local, .failed):            return ("Engine failed", .bad)
        case (.remote, .idle):             return ("Not connected", .off)
        case (.remote, .connecting):       return ("Connecting…", .busy)
        case (.remote, .ready):            return ("Connected · \(settings.remoteURL?.host ?? "server")", .ok)
        case (.remote, .failed):           return ("Offline", .bad)
        }
    }

    var appVersion: String {
        let v = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0"
        let b = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "1"
        return "\(v) (\(b))"
    }

    // MARK: - Connection

    func connectIfNeeded() async {
        guard !didAutoConnect else { return }
        didAutoConnect = true
        #if os(macOS)
        // First run: hosted providers need the user's key; Apple's on-device model needs nothing.
        if settings.engineMode == .local, settings.provider.needsKey, settings.currentKey == nil, !settings.hasSeenWelcome {
            settings.hasSeenWelcome = true
            sheet = .keyNeeded
        }
        #endif
        await connect()
        #if DEBUG
        if connection.isReady { await importLaunchFileIfAny() }
        #endif
    }

    #if DEBUG
    /// Developer/QA automation, compiled into Debug builds only — a shipped (Release) app never
    /// reads these, so it always starts empty. `CADVERT --open /path/to/part.step`
    /// (Terminal: `open -a CADVERT --args --open part.step`; Simulator:
    /// `xcrun simctl launch <udid> com.cadvert.CADVERT --open <path>` or env CADVERT_OPEN_FILE).
    func importLaunchFileIfAny() async {
        let args = CommandLine.arguments
        var path: String?
        if let i = args.firstIndex(where: { $0 == "--open" || $0 == "-open" }), i + 1 < args.count { path = args[i + 1] }
        if path == nil, let env = ProcessInfo.processInfo.environment["CADVERT_OPEN_FILE"], !env.isEmpty { path = env }
        if let path {
            await importFile(url: URL(fileURLWithPath: (path as NSString).expandingTildeInPath))
        }
        // QA / automation helpers: `--ask "question"` sends a message once the part is loaded,
        // `--show hsd|about|developers|settings` opens that sheet.
        if let i = args.firstIndex(of: "--ask"), i + 1 < args.count, session != nil {
            send(args[i + 1])
        }
        if let i = args.firstIndex(of: "--show"), i + 1 < args.count, let s = AppSheet(rawValue: args[i + 1]) {
            sheet = s
        }
    }
    #endif

    /// (Re)connects according to Settings: boots the local engine on macOS, or pings a remote server.
    func connect() async {
        connectTask?.cancel()
        let task = Task { await self.performConnect() }
        connectTask = task
        await task.value
    }

    private func performConnect() async {
        client = nil
        serverConfig = nil
        switch settings.engineMode {
        case .remote:
            guard let url = settings.remoteURL else {
                connection = .failed("Enter a server address in Settings (e.g. https://cadvert.example.com)")
                return
            }
            connection = .connecting("Connecting to \(url.host ?? url.absoluteString)…")
            let c = CadvertClient(baseURL: url)
            do {
                let cfg = try await c.fetchConfig()
                guard !Task.isCancelled else { return }
                client = c
                serverConfig = cfg
                connection = .ready
            } catch {
                connection = .failed("Could not reach \(url.absoluteString): \(error.localizedDescription)")
            }

        case .local:
            #if os(macOS)
            engine.stop()
            connection = .connecting("Locating engine…")
            let (python, trail) = await EngineLocator.resolve(customPath: settings.pythonPath)
            engineTrail = trail
            guard let python else {
                if EngineLocator.bundleHasOtherArchEngineOnly() {
                    connection = .failed("This download is for a different kind of Mac. It has no \(EngineLocator.currentArch == "x86_64" ? "Intel" : "Apple silicon") engine — please download the \(EngineLocator.currentArch == "x86_64" ? "Intel" : "Apple silicon") build of CADVERT.")
                    return
                }
                connection = .failed("No CADVERT engine found on this Mac.\n\nBundle one with scripts/bundle-engine.sh, install it to ~/.cadvert/engine, or point Settings at a Python that has `pip install cadvert[server]`.")
                return
            }
            connection = .connecting("Starting local engine…")
            do {
                let base = try await engine.start(python: python,
                                                  openAIKey: settings.trimmedOpenAIKey,
                                                  anthropicKey: settings.trimmedAnthropicKey)
                guard !Task.isCancelled else { return }
                let c = CadvertClient(baseURL: base)
                serverConfig = try await c.fetchConfig()
                client = c
                connection = .ready
            } catch {
                engineLog = engine.snapshotLog()
                connection = .failed(error.localizedDescription)
            }
            #else
            connection = .failed("Local engine is macOS-only — choose a remote server.")
            #endif
        }
    }

    func stopEngine() {
        #if os(macOS)
        engine.stop()
        #endif
        connectTask?.cancel()
        if session != nil, let client, let session { Task { await client.deleteSession(session.id) } }
    }

    func refreshEngineLog() {
        #if os(macOS)
        engineLog = engine.snapshotLog()
        #endif
    }

    // MARK: - Files

    func importFile(url: URL) async {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        do {
            let data = try Data(contentsOf: url)
            await importData(data, filename: url.lastPathComponent)
        } catch {
            showToast("Could not read \(url.lastPathComponent): \(error.localizedDescription)")
        }
    }

    /// Upload + full pipeline. Mirrors `uploadFile()` in index.html, including the staged progress text.
    func importData(_ data: Data, filename: String) async {
        guard SupportedFormat.from(filename: filename) != nil else {
            let ext = (filename as NSString).pathExtension.lowercased()
            showToast(CadvertError.unsupportedFormat(ext.isEmpty ? "?" : ext).localizedDescription)
            return
        }
        guard let client, connection.isReady else {
            showToast(connection.failureMessage.map { "Engine unavailable — \($0.split(separator: "\n").first ?? "")" } ?? CadvertError.notConnected.localizedDescription)
            if case .failed = connection { sheet = .settings }
            return
        }
        if processing != nil { return }
        if let old = session { Task { await client.deleteSession(old.id) } }

        lastImport = (data, filename)
        resetSession()
        showPartSheet = false
        processing = ProcessingState()

        let ticker = Task { [weak self] in
            var elapsed = 0
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                if Task.isCancelled { break }
                elapsed += 1
                guard let self else { break }
                self.processing?.elapsed = elapsed
                self.processing?.stage = ProcessingStages.label(elapsed: elapsed)
                self.processing?.percent = ProcessingStages.percent(elapsed: elapsed)
            }
        }
        defer { ticker.cancel() }

        do {
            let r = try await client.convert(fileData: data, filename: filename)
            processing?.stage = "Finalising document…"
            processing?.percent = 98
            let elapsed = processing?.elapsed ?? 0
            session = PartSession(r)
            transcript = [
                .system(id: UUID(), text: session!.systemSummaryLine),
                .chips(id: UUID(), questions: SuggestedQuestions.all),
            ]
            messagesSent = 0
            processing = nil
            doneLabel = "Done in \(elapsed)s"
            let label = doneLabel
            Task { [weak self] in
                try? await Task.sleep(nanoseconds: 2_500_000_000)
                if self?.doneLabel == label { self?.doneLabel = nil }
            }
            if let pending = pendingMessage { pendingMessage = nil; send(pending) }
        } catch let e as CadvertError {
            processing = nil
            switch e {
            default:            showToast(e.localizedDescription)
            }
        } catch {
            processing = nil
            showToast(error.localizedDescription)
        }
    }

    /// Finder / Files drag-and-drop.
    func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first else { return false }
        if provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
            provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier) { item, _ in
                var url: URL?
                if let data = item as? Data { url = URL(dataRepresentation: data, relativeTo: nil) }
                else if let u = item as? URL { url = u }
                guard let url else { return }
                Task { @MainActor in await self.importFile(url: url) }
            }
            return true
        }
        for type in SupportedFormat.contentTypes where provider.hasItemConformingToTypeIdentifier(type.identifier) {
            let suggested = provider.suggestedName
            provider.loadFileRepresentation(forTypeIdentifier: type.identifier) { tmp, _ in
                guard let tmp, let data = try? Data(contentsOf: tmp) else { return }
                var name = tmp.lastPathComponent
                if SupportedFormat.from(filename: name) == nil, let s = suggested { name = s }
                Task { @MainActor in await self.importData(data, filename: name) }
            }
            return true
        }
        return false
    }

    func newSession() {
        if let client, let s = session { Task { await client.deleteSession(s.id) } }
        resetSession()
        lastImport = nil
    }

    private func resetSession() {
        appleChat.reset()
        session = nil
        transcript = []
        messagesSent = 0
        streaming = false
        draft = ""
        doneLabel = nil
        lightboxURL = nil
    }

    // MARK: - Chat

    func sendDraft() { send(draft) }

    /// Mirrors `_sendText()` in index.html: optimistic user bubble, typing dots, SSE stream with tool indicators.
    func send(_ rawText: String) {
        let text = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        guard let session else { showToast("Upload a CAD file first"); return }
        guard !streaming, let client else { return }

        let provider = settings.provider

        // The offline Mac engine can only call a hosted provider with the user's own key (passed at launch).
        if provider.needsKey, settings.engineMode == .local, settings.currentKey == nil {
            pendingMessage = text
            sheet = .keyNeeded
            return
        }
        if provider == .apple, !AppleIntelligence.status.available {
            showToast(AppleIntelligence.status.detail)
            sheet = .settings
            return
        }

        pendingMessage = nil
        draft = ""
        hideChips()

        let userMsg = ChatMessage.user(text)
        let typing = ChatMessage.typing()
        transcript.append(.message(userMsg))
        let history = serverHistory()
        transcript.append(.message(typing))
        streaming = true

        let stream: AsyncThrowingStream<ChatEvent, Error>
        if provider == .apple {
            stream = appleChat.stream(question: text, part: session, client: client)
        } else {
            stream = client.chat(sessionId: session.id, history: history, provider: provider,
                                 apiKey: settings.currentKey, model: settings.currentModel)
        }
        Task { await self.consume(stream, text: text, userId: userMsg.id, typingId: typing.id) }
    }

    private func consume(_ stream: AsyncThrowingStream<ChatEvent, Error>, text: String, userId: UUID, typingId: UUID) async {
        var full = ""
        defer { streaming = false }
        do {
            for try await event in stream {
                switch event {
                case .toolCall(let name):
                    update(typingId) { $0.isTyping = false; $0.toolLabel = ToolLabels.label(for: name) }
                case .content(let chunk):
                    full += chunk
                    let snapshot = full
                    update(typingId) { $0.isTyping = false; $0.toolLabel = nil; $0.text = snapshot }
                case .error(let m):
                    throw CadvertError.stream(m)
                case .done:
                    break
                }
            }
            messagesSent += 1
            update(typingId) { $0.isTyping = false; $0.toolLabel = nil; $0.text = full }
            if full.isEmpty { remove(typingId) }
        } catch let e as CadvertError {
            switch e {
            case .apiKeyRequired:
                remove(typingId); remove(userId); pendingMessage = text; sheet = .keyNeeded
            default:
                failMessage(userId: userId, typingId: typingId, message: e.localizedDescription)
            }
        } catch {
            failMessage(userId: userId, typingId: typingId, message: error.localizedDescription)
        }
    }

    private func failMessage(userId: UUID, typingId: UUID, message: String) {
        remove(typingId)
        update(userId) { $0.isError = true }   // dropped from history, like chatHistory.pop()
        transcript.append(.message(ChatMessage(role: .assistant, text: "Error: \(message)", isError: true)))
    }

    /// `chatHistory` as the server expects it: user/assistant turns, no typing rows, no failed turns.
    private func serverHistory() -> [[String: String]] {
        transcript.compactMap { item in
            guard case .message(let m) = item, !m.isTyping, !m.isError, !m.text.isEmpty else { return nil }
            return ["role": m.role.rawValue, "content": m.text]
        }
    }

    private func hideChips() {
        transcript.removeAll { if case .chips = $0 { return true } else { return false } }
    }

    private func update(_ id: UUID, _ change: (inout ChatMessage) -> Void) {
        guard let idx = transcript.firstIndex(where: { $0.id == id }), case .message(var m) = transcript[idx] else { return }
        change(&m)
        transcript[idx] = .message(m)
    }

    private func remove(_ id: UUID) {
        transcript.removeAll { $0.id == id }
    }

    // MARK: - Keys

    /// Save a provider key. The local engine reads keys from its environment, so it is restarted
    /// (and the last part re-processed) — then any pending message is sent.
    func saveAPIKey(_ key: String, for provider: AIProvider) async {
        let trimmed = key.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("sk-") else { showToast("Enter a valid \(provider.keyVendor) key (starts with sk-)"); return }
        settings.setKey(trimmed, for: provider)
        if settings.provider != provider, provider.needsKey { settings.provider = provider }
        sheet = nil
        if settings.engineMode == .local {
            let reimport = lastImport
            await connect()
            if connection.isReady, let (data, name) = reimport {
                await importData(data, filename: name)   // resends pendingMessage when done
            } else if !connection.isReady {
                showToast(connection.failureMessage ?? "Engine failed to restart")
            }
        } else {
            showToast("Key saved — continuing with your API key", success: true)
            if let pending = pendingMessage { pendingMessage = nil; send(pending) }
        }
    }

    // MARK: - HSD / images / misc

    func copyHSD() {
        guard let hsd = session?.hsd else { return }
        #if os(macOS)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(hsd, forType: .string)
        #else
        UIPasteboard.general.string = hsd
        #endif
        showToast("HSD copied to clipboard", success: true)
    }

    func copyToClipboard(_ s: String, toast: String? = nil) {
        #if os(macOS)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(s, forType: .string)
        #else
        UIPasteboard.general.string = s
        #endif
        if let toast { showToast(toast, success: true) }
    }

    func imageURL(for image: ViewImage) -> URL? { client?.imageURL(for: image) }

    func showToast(_ text: String, success: Bool = false) {
        toast = Toast(text: text, success: success)
        toastTask?.cancel()
        toastTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 4_000_000_000)
            if !Task.isCancelled { self?.toast = nil }
        }
    }
}

/// Plain-text document used by the "Save HSD…" exporter.
struct HSDDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.plainText] }
    var text: String
    init(text: String) { self.text = text }
    init(configuration: ReadConfiguration) throws {
        text = String(decoding: configuration.file.regularFileContents ?? Data(), as: UTF8.self)
    }
    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: Data(text.utf8))
    }
}

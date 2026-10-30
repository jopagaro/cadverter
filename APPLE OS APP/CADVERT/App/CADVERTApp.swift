import SwiftUI

@main
struct CADVERTApp: App {
    #if os(macOS)
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    #endif
    private let model = AppModel.shared

    private var mainContent: some View {
        RootView()
            .environment(model)
            .onOpenURL { url in
                Task { await model.importFile(url: url) }
            }
    }

    var body: some Scene {
        #if os(macOS)
        WindowGroup { mainContent }
            .defaultSize(width: 1180, height: 760)
            .commands { AppCommands(model: model) }

        Settings {
            SettingsView(embedded: true)
                .environment(model)
                .frame(width: 560)
        }
        #else
        WindowGroup { mainContent }
        #endif
    }
}

#if os(macOS)
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var sigterm: DispatchSourceSignal?

    func applicationDidFinishLaunching(_ notification: Notification) {
        #if DEBUG
        if Self.renderReplyIfAsked() { NSApp.terminate(nil); return }
        #endif
        // `kill <pid>` / logout: quit gracefully so the engine child is stopped too.
        signal(SIGTERM, SIG_IGN)
        let src = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
        src.setEventHandler { NSApp.terminate(nil) }
        src.resume()
        sigterm = src
    }

    func application(_ application: NSApplication, open urls: [URL]) {
        for url in urls {
            Task { await AppModel.shared.importFile(url: url) }
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        AppModel.shared.stopEngine()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    #if DEBUG
    /// `CADVERT --render-reply reply.md out.png [dark|light]` draws a reply's content to a PNG
    /// and quits: a check on the renderer that needs no window, engine or provider key.
    static func renderReplyIfAsked() -> Bool {
        let args = CommandLine.arguments
        guard let i = args.firstIndex(of: "--render-reply"), i + 2 < args.count else { return false }
        guard let text = try? String(contentsOfFile: args[i + 1], encoding: .utf8) else { return true }
        let dark = i + 3 >= args.count || args[i + 3] != "light"
        let palette: Palette = dark ? .dark : .light
        let view = MarkdownText(text: text, isUser: false)
            .environment(\.palette, palette)
            .environment(\.colorScheme, dark ? .dark : .light)
            .padding(16)
            .frame(width: 760, alignment: .leading)
            .background(palette.surface)
        let renderer = ImageRenderer(content: view)
        renderer.scale = 2
        if let tiff = renderer.nsImage?.tiffRepresentation,
           let png = NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:]) {
            try? png.write(to: URL(fileURLWithPath: args[i + 2]))
        }
        return true
    }
    #endif
}

struct AppCommands: Commands {
    let model: AppModel

    var body: some Commands {
        CommandGroup(replacing: .newItem) {
            Button("Open CAD File…") { model.showFileImporter = true }
                .keyboardShortcut("o", modifiers: .command)
            Button("New Session") { model.newSession() }
                .keyboardShortcut("n", modifiers: .command)
        }
        CommandGroup(after: .importExport) {
            Button("Export HSD…") { model.showHSDExporter = true }
                .keyboardShortcut("e", modifiers: [.command, .shift])
                .disabled(model.session == nil)
            Menu("Export Part") { PartExportItems(model: model) }
                .disabled(model.session == nil || model.exporting != nil)
            Button("Copy HSD") { model.copyHSD() }
                .keyboardShortcut("c", modifiers: [.command, .shift])
                .disabled(model.session == nil)
            Button("View HSD") { model.sheet = .hsd }
                .keyboardShortcut("h", modifiers: [.command, .shift])
                .disabled(model.session == nil)
        }
        CommandGroup(replacing: .appInfo) {
            Button("About CADVERT") { model.sheet = .about }
        }
        CommandMenu("Engine") {
            Button("Restart Engine") { Task { await model.connect() } }
                .keyboardShortcut("r", modifiers: [.command, .shift])
            Button("Engine Log…") { model.refreshEngineLog(); model.sheet = .engineLog }
        }
        CommandGroup(after: .toolbar) {
            Picker("Theme", selection: Binding(get: { model.settings.theme }, set: { model.settings.theme = $0 })) {
                ForEach(ThemePreference.allCases) { t in Text(t.label).tag(t) }
            }
        }
        CommandGroup(replacing: .help) {
            Button("For Developers (pip install cadvert)") { model.sheet = .developers }
            Link("cadvert on PyPI", destination: URL(string: "https://pypi.org/project/cadvert/")!)
            Divider()
            Button("Acknowledgements") { model.sheet = .acknowledgements }
        }
    }
}
#endif

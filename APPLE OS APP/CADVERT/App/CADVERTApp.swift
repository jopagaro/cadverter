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
        }
    }
}
#endif

import SwiftUI
import UniformTypeIdentifiers

/// Header + (sidebar | main) — the `.layout` grid of index.html. Collapses to a single
/// column on iPhone, where the sidebar becomes a "Part" sheet.
struct RootView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.colorScheme) private var scheme
    #if os(iOS)
    @Environment(\.horizontalSizeClass) private var hSize
    #endif

    private var isCompact: Bool {
        #if os(iOS)
        return hSize == .compact
        #else
        return false
        #endif
    }

    var body: some View {
        @Bindable var model = model
        let p = Palette.forScheme(scheme)

        ZStack {
            p.bg.ignoresSafeArea()

            VStack(spacing: 0) {
                HeaderBar(isCompact: isCompact)
                Hairline()
                if isCompact {
                    MainArea(isCompact: true)
                } else {
                    HStack(spacing: 0) {
                        SidebarView()
                            .frame(width: 232)
                        Hairline(vertical: true)
                        MainArea(isCompact: false)
                    }
                }
            }

            if let proc = model.processing {
                ProcessingOverlay(state: proc)
                    .transition(.opacity)
            }
            if let url = model.lightboxURL {
                LightboxView(url: url) { model.lightboxURL = nil }
                    .transition(.opacity)
            }
            if let toast = model.toast {
                VStack {
                    Spacer()
                    ToastView(toast: toast)
                        .padding(.bottom, 72)
                }
                .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .animation(.easeOut(duration: 0.2), value: model.toast)
        .animation(.easeOut(duration: 0.2), value: model.processing == nil)
        .environment(\.palette, p)
        .preferredColorScheme(model.settings.theme.colorScheme)
        .tint(p.accent)
        .sheet(item: $model.sheet) { sheet in
            SheetHost(sheet: sheet)
                .environment(\.palette, p)
                .environment(model)
        }
        .sheet(isPresented: $model.showPartSheet) {
            PartSheet()
                .environment(\.palette, p)
                .environment(model)
        }
        .fileImporter(isPresented: $model.showFileImporter,
                      allowedContentTypes: SupportedFormat.contentTypes + [.data],
                      allowsMultipleSelection: false) { result in
            switch result {
            case .success(let urls):
                if let url = urls.first { Task { await model.importFile(url: url) } }
            case .failure(let error):
                model.showToast(error.localizedDescription)
            }
        }
        .fileExporter(isPresented: $model.showHSDExporter,
                      document: HSDDocument(text: model.session?.hsd ?? ""),
                      contentType: .plainText,
                      defaultFilename: "\(model.session?.stem ?? "part").hsd.txt") { result in
            if case .failure(let error) = result { model.showToast(error.localizedDescription) }
            else { model.showToast("HSD saved", success: true) }
        }
        .onDrop(of: [.fileURL] + SupportedFormat.contentTypes, isTargeted: $model.isDropTargeted) { providers in
            model.handleDrop(providers)
        }
        .task { await model.connectIfNeeded() }
        #if os(macOS)
        .frame(minWidth: 940, minHeight: 600)
        #endif
    }
}

/// `.main` — views strip + chat + input bar.
struct MainArea: View {
    var isCompact: Bool
    @Environment(\.palette) private var p

    var body: some View {
        VStack(spacing: 0) {
            ViewsStrip()
            Hairline()
            ChatView(isCompact: isCompact)
            Hairline()
            InputBar()
        }
        .background(p.bg)
    }
}

/// Routes `AppSheet` cases to their views.
struct SheetHost: View {
    var sheet: AppSheet
    @Environment(AppModel.self) private var model
    @Environment(\.palette) private var p

    var body: some View {
        Group {
            switch sheet {
            case .hsd:         HSDSheet()
            case .about:       AboutView(section: .about)
            case .developers:  AboutView(section: .developers)
            case .settings:    SettingsView(embedded: false)
            case .keyNeeded:   KeyNeededView()
            case .engineLog:   EngineLogView()
            }
        }
        .background(p.surface)
        .preferredColorScheme(model.settings.theme.colorScheme)
    }
}

/// iPhone: the sidebar (upload + part info) presented as a sheet.
struct PartSheet: View {
    @Environment(AppModel.self) private var model
    @Environment(\.palette) private var p

    var body: some View {
        NavigationStack {
            SidebarView()
                .navigationTitle("Part")
                #if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
                #endif
                .toolbar {
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Done") { model.showPartSheet = false }
                    }
                }
        }
        .background(p.surface)
    }
}

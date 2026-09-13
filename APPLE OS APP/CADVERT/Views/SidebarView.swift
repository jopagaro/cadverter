import SwiftUI

/// `<aside class="sidebar">` — upload dropzone, progress, part info card.
struct SidebarView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.palette) private var p

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                VStack(alignment: .leading, spacing: 8) {
                    SectionLabel(text: "Upload File")
                    DropzoneView()
                }
                if let done = model.doneLabel {
                    ProgressRow(label: done, percent: 100)
                }
                if let s = model.session {
                    PartInfoCard(session: s)
                }
                if let failure = model.connection.failureMessage {
                    EngineFailureCard(message: failure)
                }
            }
            .padding(14)
        }
        .background(p.surface)
    }
}

/// `.dropzone` — dashed drop target that also opens the file picker.
struct DropzoneView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.palette) private var p
    @State private var hover = false

    private var highlighted: Bool { hover || model.isDropTargeted }

    var body: some View {
        Button { model.showFileImporter = true } label: {
            VStack(spacing: 0) {
                Text("↑")
                    .font(.system(size: 20, design: .monospaced))
                    .opacity(0.3)
                    .padding(.bottom, 8)
                Text("Drop file or browse")
                    .typo(12, .medium)
                    .foregroundStyle(p.text)
                    .padding(.bottom, 4)
                Text("Any supported CAD format")
                    .typo(10, .medium)
                    .foregroundStyle(p.textDim)
                    .padding(.bottom, 10)
                FlowLayout(spacing: 4, alignment: .center) {
                    ForEach(SupportedFormat.allCases, id: \.rawValue) { f in
                        FormatTag(ext: f.rawValue, full: f.isFullAnalysis)
                    }
                }
                .padding(.bottom, 8)
                HStack(spacing: 4) {
                    Text("■").foregroundStyle(p.success)
                    Text("full analysis")
                    Text(" ")
                    Text("■").foregroundStyle(p.textMuted)
                    Text("mesh only")
                }
                .typo(9, .medium)
                .foregroundStyle(p.textDim)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 22)
            .padding(.horizontal, 12)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(highlighted ? p.accentSoft : .clear)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(highlighted ? p.accent : p.border, style: StrokeStyle(lineWidth: 1.5, dash: [5, 3]))
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hover = $0 }
        .animation(.easeOut(duration: 0.15), value: highlighted)
        .accessibilityLabel("Choose a CAD file")
    }
}

/// `.progress-wrap` — thin bar + label ("Done in 12s").
struct ProgressRow: View {
    var label: String
    var percent: Double
    @Environment(\.palette) private var p

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 2).fill(p.surfaceAlt)
                    RoundedRectangle(cornerRadius: 2).fill(p.accent)
                        .frame(width: geo.size.width * percent / 100)
                }
            }
            .frame(height: 3)
            Text(label).typo(10, .medium).foregroundStyle(p.textMuted)
        }
    }
}

/// `.info-card` — File / Format / Units / Faces… + detected features + "View HSD →".
struct PartInfoCard: View {
    var session: PartSession
    @Environment(AppModel.self) private var model
    @Environment(\.palette) private var p

    var body: some View {
        let s = session.summary
        VStack(alignment: .leading, spacing: 0) {
            SectionLabel(text: "Part Info").padding(.bottom, 10)

            InfoRow(key: "File") { Text(session.filename).lineLimit(1).truncationMode(.middle) }
            InfoRow(key: "Format") { Badge(text: session.format, style: session.isMesh ? .orange : .blue) }
            if let schema = s.schema, !schema.isEmpty { InfoRow(key: "Schema") { Text(schema) } }
            InfoRow(key: "Units") { Text(session.units) }
            if !session.isMesh {
                if let f = s.faces { InfoRow(key: "Faces") { Text("\(f)") } }
                if let e = s.edges { InfoRow(key: "Edges") { Text("\(e)") } }
                if let b = s.bodies { InfoRow(key: "Bodies") { Text("\(b)") } }
                if let g = s.gdtCount, g > 0 { InfoRow(key: "GD&T") { Text("\(g) annotations") } }
            } else {
                InfoRow(key: "Triangles") { Text((s.triangles ?? 0).formatted()) }
                InfoRow(key: "Analysis") { Badge(text: "mesh only", style: .orange) }
            }

            let feats = s.sortedFeatures
            if !feats.isEmpty {
                VStack(alignment: .leading, spacing: 2) {
                    SectionLabel(text: "Detected Features").padding(.bottom, 6)
                    ForEach(feats, id: \.name) { f in
                        HStack {
                            Text(f.name.replacingOccurrences(of: "_", with: " "))
                                .typo(11, .medium)
                                .foregroundStyle(p.textMuted)
                            Spacer()
                            Text("\(f.count)")
                                .typo(10, .semibold, mono: true)
                                .foregroundStyle(p.text)
                        }
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(p.bg, in: RoundedRectangle(cornerRadius: 5, style: .continuous))
                    }
                }
                .padding(.top, 10)
            }

            HStack(spacing: 10) {
                Button("View HSD →") { model.sheet = .hsd }
                    .buttonStyle(PlainTextButtonStyle())
                    .typo(10, .medium)
                    .foregroundStyle(p.isDark ? p.accent : p.textMuted)
                Button("Copy") { model.copyHSD() }
                    .buttonStyle(PlainTextButtonStyle())
                    .typo(10, .medium)
                    .foregroundStyle(p.textDim)
                #if os(macOS)
                Button("Save…") { model.showHSDExporter = true }
                    .buttonStyle(PlainTextButtonStyle())
                    .typo(10, .medium)
                    .foregroundStyle(p.textDim)
                #endif
                Spacer()
                Button("New") { model.newSession() }
                    .buttonStyle(PlainTextButtonStyle())
                    .typo(10, .medium)
                    .foregroundStyle(p.textDim)
                    .help("Start a new session")
            }
            .padding(.top, 12)
        }
        .padding(12)
        .background(p.surfaceAlt, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
}

struct InfoRow<Value: View>: View {
    var key: String
    @ViewBuilder var value: Value
    @Environment(\.palette) private var p

    var body: some View {
        HStack {
            Text(key).typo(11, .medium).foregroundStyle(p.textMuted)
            Spacer(minLength: 8)
            value
                .typo(10, .semibold, mono: true)
                .foregroundStyle(p.text)
        }
        .padding(.vertical, 3)
    }
}

/// Shown in the sidebar when the engine could not start / server is unreachable.
struct EngineFailureCard: View {
    var message: String
    @Environment(AppModel.self) private var model
    @Environment(\.palette) private var p

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Circle().fill(p.danger).frame(width: 6, height: 6)
                SectionLabel(text: "Engine unavailable")
            }
            Text(message)
                .typo(10.5, .medium)
                .foregroundStyle(p.textMuted)
                .lineSpacing(2)
                .fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 8) {
                Button("Retry") { Task { await model.connect() } }.buttonStyle(GhostButtonStyle())
                Button("Settings") { model.sheet = .settings }.buttonStyle(GhostButtonStyle())
            }
        }
        .padding(12)
        .background(p.surfaceAlt, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
}

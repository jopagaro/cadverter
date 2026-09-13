import SwiftUI

/// `.processing-overlay` — spinner, stage, big seconds counter, progress bar.
struct ProcessingOverlay: View {
    var state: ProcessingState
    @Environment(\.palette) private var p

    var body: some View {
        ZStack {
            Color.black.opacity(0.6).ignoresSafeArea()
            VStack(spacing: 0) {
                SpinnerRing().padding(.bottom, 20)
                Text("Analyzing CAD file...")
                    .typo(14, .semibold).foregroundStyle(p.text).padding(.bottom, 6)
                Text(state.stage)
                    .typo(12, .medium).foregroundStyle(p.accent)
                    .frame(minHeight: 18).padding(.bottom, 14)
                Text("\(state.elapsed)")
                    .font(.system(size: 28, weight: .bold, design: .default).monospacedDigit())
                    .tracking(-1)
                    .foregroundStyle(p.text)
                Text("seconds elapsed").typo(10, .medium).foregroundStyle(p.textDim).padding(.top, 2)
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 4).fill(p.surfaceAlt)
                        RoundedRectangle(cornerRadius: 4).fill(p.accent)
                            .frame(width: geo.size.width * state.percent / 100)
                            .animation(.easeOut(duration: 0.5), value: state.percent)
                    }
                }
                .frame(height: 3)
                .padding(.top, 18)
            }
            .padding(.vertical, 32)
            .padding(.horizontal, 40)
            .frame(minWidth: 300)
            .background(p.surface, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(p.border, lineWidth: 1))
        }
    }
}

/// `.error-toast`
struct ToastView: View {
    var toast: Toast
    @Environment(\.palette) private var p

    var body: some View {
        Text(toast.text)
            .typo(12, .medium)
            .foregroundStyle(.white)
            .multilineTextAlignment(.center)
            .padding(.horizontal, 20)
            .padding(.vertical, 10)
            .background(toast.success ? p.success : p.danger, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            .frame(maxWidth: 420)
            .shadow(color: .black.opacity(0.15), radius: 8, y: 3)
    }
}

/// `.modal-box` for the HSD text — header, mono body, Copy / Save footer.
struct HSDSheet: View {
    @Environment(AppModel.self) private var model
    @Environment(\.palette) private var p

    private static let displayCap = 400_000

    var body: some View {
        let hsd = model.session?.hsd ?? ""
        let shown = hsd.count > Self.displayCap ? String(hsd.prefix(Self.displayCap)) : hsd
        VStack(spacing: 0) {
            HStack {
                Text("Hierarchical Spatial Document").typo(13, .semibold).foregroundStyle(p.text)
                Spacer()
                CloseCircle { model.sheet = nil }
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 14)
            Hairline()
            ScrollView([.vertical, .horizontal]) {
                VStack(alignment: .leading, spacing: 12) {
                    Text(shown)
                        .typo(11.5, .regular, mono: true)
                        .lineSpacing(5)
                        .foregroundStyle(p.textMuted)
                        .textSelection(.enabled)
                    if shown.count < hsd.count {
                        Text("Showing the first \(Self.displayCap.formatted()) characters — use Save… for the full document.")
                            .typo(10, .medium).foregroundStyle(p.textDim)
                    }
                }
                .padding(20)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            Hairline()
            HStack(spacing: 8) {
                Text("\(hsd.count.formatted()) chars").typo(10, .medium).foregroundStyle(p.textDim)
                Spacer()
                Button("Copy") { model.copyHSD() }.buttonStyle(GhostButtonStyle())
                Button("Save…") { model.sheet = nil; model.showHSDExporter = true }.buttonStyle(PrimaryButtonStyle(compact: true))
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 10)
        }
        .background(p.surface)
        #if os(macOS)
        .frame(width: 780, height: 640)
        #endif
    }
}

/// macOS: live output of the local engine process.
struct EngineLogView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.palette) private var p

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Engine log").typo(13, .semibold).foregroundStyle(p.text)
                Spacer()
                Button("Refresh") { model.refreshEngineLog() }.buttonStyle(GhostButtonStyle())
                Button("Copy") { model.copyToClipboard(model.engineLog + "\n\n" + model.engineTrail, toast: "Log copied") }.buttonStyle(GhostButtonStyle())
                CloseCircle { model.sheet = nil }
            }
            .padding(.horizontal, 20).padding(.vertical, 14)
            Hairline()
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    if !model.engineTrail.isEmpty {
                        SectionLabel(text: "Engine search")
                        Text(model.engineTrail).typo(10.5, .regular, mono: true).foregroundStyle(p.textMuted)
                            .textSelection(.enabled)
                    }
                    SectionLabel(text: "Process output")
                    Text(model.engineLog.isEmpty ? "(no output yet)" : model.engineLog)
                        .typo(10.5, .regular, mono: true).foregroundStyle(p.textMuted)
                        .textSelection(.enabled)
                }
                .padding(20)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .background(p.surface)
        .onAppear { model.refreshEngineLog() }
        #if os(macOS)
        .frame(width: 720, height: 520)
        #endif
    }
}

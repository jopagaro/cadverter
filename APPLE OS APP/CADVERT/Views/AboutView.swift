import SwiftUI

/// Native version of `/about` (and the developer pointer): the editorial page, same copy.
struct AboutView: View {
    enum Section { case about, developers }
    var section: Section
    @Environment(AppModel.self) private var model
    @Environment(\.palette) private var p

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                LogoWrap()
                Text(section == .about ? "About" : "For Developers")
                    .typo(10, .bold).tracking(0.6)
                    .foregroundStyle(p.textMuted)
                    .padding(.horizontal, 7).padding(.vertical, 2)
                    .background(p.surfaceAlt, in: RoundedRectangle(cornerRadius: 5, style: .continuous))
                Spacer()
                CloseCircle { model.sheet = nil }
            }
            .padding(.horizontal, 20).padding(.vertical, 12)
            Hairline()

            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 34) {
                        // Hero
                        VStack(alignment: .leading, spacing: 12) {
                            Text("cadvert · the missing layer between CAD and AI")
                                .typo(10, .bold).tracking(0.8).textCase(.uppercase).foregroundStyle(p.textMuted)
                            Text("CADVERT turns CAD files into something an AI can actually read.")
                                .font(.system(size: 26 * Typo.scale, weight: .bold)).tracking(-0.6)
                                .foregroundStyle(p.text).fixedSize(horizontal: false, vertical: true)
                            Text("Upload a part, ask anything about it — dimensions, features, manufacturability, tolerances — and get answers from the exact geometry the engineer authored, not a guess from a blurry render.")
                                .typo(14, .medium).foregroundStyle(p.textMuted).lineSpacing(4)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        .id(Section.about)

                        // What is CADVERT
                        VStack(alignment: .leading, spacing: 12) {
                            H2("What is CADVERT?")
                            Para("A STEP or IGES file is built for engineering software, not for language models. Upload one to an AI assistant and it chokes — the file is a dense graph of surface equations and topology, not a document. Show it a screenshot instead and the model can see the shape, but it can't tell you the part is exactly 8.000 mm thick or that two holes sit 30.000 mm apart. In engineering, the number is the whole point.")
                            Para("CADVERT bridges that gap. It reads the CAD file directly, extracts the exact geometry — dimensions, holes, fillets, wall thicknesses, tolerances — and writes it out as clean, structured text any LLM can reason over.")
                            Para("The key word is **exact**. CADVERT doesn't approximate the part into a mesh of triangles the way most tools do. It reads the true analytical geometry straight from the source, so every dimension it reports is the value that was actually in the file — not a measurement, not an estimate.")
                            Text("Instead of *\"a bracket with some holes,\"* your model gets this:")
                                .typo(12.5, .medium).foregroundStyle(p.textMuted).padding(.top, 4)
                            Sample("""
                            THROUGH HOLE  d=8.000mm  depth=20.000mm  [5/16"]
                            PATTERN       2× through hole  d=8.000mm
                            Wall thickness:  bore → outer surface  = 11.000 mm
                            Hole-to-hole center distance           = 30.000 mm
                            """)
                            Para("It recognized the through holes, matched the diameter to a standard 5/16\" drill, spotted the pattern, and computed the exact wall thickness — the dimensions a machinist or a DFM reviewer actually signs off on.")
                        }

                        // How to use
                        VStack(alignment: .leading, spacing: 12) {
                            H2("How to use it")
                            Para("Two ways to use CADVERT, depending on who you are.")
                            H3("If you just want answers about a part")
                            Para("Open your file — STEP, IGES, BREP, STL, or OBJ — and ask. *\"How thick is this wall?\"* *\"What size tap do these holes need?\"* *\"Is this part suitable for CNC machining?\"* CADVERT reads the exact geometry and the AI answers from real numbers, not guesses.")
                            Para("On the Mac the whole analysis runs locally and offline. On iPad and iPhone the app talks to a CADVERT server.")

                            H3("If you're a developer building your own tool").id(Section.developers)
                            InstallStrip()
                            Para("One call takes a CAD file to AI-ready context you can drop into any model. Some of what people build:")
                            UseTable()
                            Sample("""
                            import cadvert

                            result = cadvert.analyze("bracket.step")
                            print(result.to_text())     # LLM-ready structured document
                            result.to_dict()            # JSON for quoting / CI / DBs
                            result.to_graph()           # networkx face-adjacency graph
                            result.to_points(2048)      # (N,3) surface point cloud
                            """)
                            Para("Whatever you build, the geometry underneath is exact. CADVERT does the translation; your model does the reasoning. The full API reference lives on the PyPI project page and the web app's developer docs.")
                            HStack(spacing: 10) {
                                Link("cadvert on PyPI →", destination: URL(string: "https://pypi.org/project/cadvert/")!)
                                    .typo(12, .semibold).foregroundStyle(p.accent)
                            }
                        }

                        // Closing
                        VStack(alignment: .leading, spacing: 12) {
                            Hairline()
                            Para("Give it a part — open a CAD file and start asking questions, or install the library and build on it.")
                            HStack(spacing: 10) {
                                Button("Open a CAD file →") { model.sheet = nil; model.showFileImporter = true }
                                    .buttonStyle(PrimaryButtonStyle())
                            }
                        }
                        .padding(.bottom, 20)
                    }
                    .padding(.horizontal, 28)
                    .padding(.vertical, 28)
                    .frame(maxWidth: 720, alignment: .leading)
                    .frame(maxWidth: .infinity)
                }
                .onAppear {
                    if section == .developers {
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) { proxy.scrollTo(Section.developers, anchor: .top) }
                    }
                }
            }
        }
        .background(p.surface)
        #if os(macOS)
        .frame(width: 780, height: 680)
        #endif
    }

    // MARK: Typographic helpers

    private func H2(_ s: String) -> some View {
        Text(s).font(.system(size: 18 * Typo.scale, weight: .bold)).tracking(-0.3).foregroundStyle(p.text)
    }
    private func H3(_ s: String) -> some View {
        Text(s).typo(13.5, .semibold).foregroundStyle(p.text).padding(.top, 6)
    }
    private func Para(_ s: String) -> some View {
        Text((try? AttributedString(markdown: s, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))) ?? AttributedString(s))
            .typo(13, .medium).foregroundStyle(p.textMuted).lineSpacing(5)
            .fixedSize(horizontal: false, vertical: true)
    }
    private func Sample(_ s: String) -> some View {
        ScrollView(.horizontal, showsIndicators: false) {
            Text(s).typo(11.5, .regular, mono: true).foregroundStyle(p.text).lineSpacing(4).padding(14)
        }
        .background(p.surfaceAlt, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).stroke(p.border, lineWidth: 1))
    }
}

/// `.install-strip` — `pip install cadvert` + Copy.
struct InstallStrip: View {
    @Environment(AppModel.self) private var model
    @Environment(\.palette) private var p
    @State private var copied = false

    var body: some View {
        HStack {
            Text("pip install cadvert").typo(12.5, .medium, mono: true).foregroundStyle(p.text)
            Spacer()
            Button(copied ? "Copied!" : "Copy") {
                model.copyToClipboard("pip install cadvert")
                copied = true
                DispatchQueue.main.asyncAfter(deadline: .now() + 2) { copied = false }
            }
            .buttonStyle(GhostButtonStyle())
        }
        .padding(.horizontal, 14).padding(.vertical, 10)
        .background(p.surfaceAlt, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).stroke(p.border, lineWidth: 1))
    }
}

/// `.use-table`
struct UseTable: View {
    @Environment(\.palette) private var p
    private let rows: [(String, String)] = [
        ("AI copilot for CAD", "Register CADVERT as a tool in a Claude or GPT agent — it answers questions about real parts using real dimensions."),
        ("Manufacturability checks", "Run it in CI to flag thin walls, holes too close to an edge, or undercuts — before a human opens the file."),
        ("Instant quoting", "A customer emails a STEP file; your system reads the features and drafts a quote automatically."),
        ("Design search & BOMs", "Extract a structured feature and dimension list from any part for search, tagging, or a bill of materials."),
    ]

    var body: some View {
        VStack(spacing: 0) {
            ForEach(Array(rows.enumerated()), id: \.offset) { i, row in
                HStack(alignment: .top, spacing: 14) {
                    Text(row.0).typo(12, .semibold).foregroundStyle(p.text).frame(width: 150, alignment: .leading)
                    Text(row.1).typo(12, .medium).foregroundStyle(p.textMuted).lineSpacing(3)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(.vertical, 10)
                if i < rows.count - 1 { Hairline() }
            }
        }
    }
}

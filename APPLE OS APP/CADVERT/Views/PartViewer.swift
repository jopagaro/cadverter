import SwiftUI
import SceneKit
#if os(macOS)
import AppKit
#else
import UIKit
#endif

/// The interactive 3D view above the chat: rotate, zoom and pan the part, jump to the
/// standard views, and click a face to see what it is — and ask about it.
struct ViewerPanel: View {
    @Environment(AppModel.self) private var model
    @Environment(\.palette) private var p
    @AppStorage("viewerShowGrid") private var showGrid = false
    @AppStorage("viewerShowDimensions") private var showDimensions = false
    @State private var camera = CameraState()

    var body: some View {
        @Bindable var model = model
        ZStack {
            ModelViewer(mesh: model.mesh, highlighted: model.highlightedFaces, command: model.viewCommand,
                        dark: p.isDark, showGrid: showGrid, showDimensions: showDimensions,
                        snapshotRequest: model.snapshotRequest, camera: camera,
                        onSnapshot: model.snapshotReady) { model.pickFace($0) }

            switch model.meshState {
            case .loading:
                VStack(spacing: 10) {
                    SpinnerRing(size: 22)
                    Text("Building 3D view…").typo(11, .medium).foregroundStyle(p.textMuted)
                }
            case .failed(let message):
                VStack(spacing: 8) {
                    Text("Couldn't build the 3D view").typo(12, .semibold).foregroundStyle(p.text)
                    Text(message).typo(10.5, .medium).foregroundStyle(p.textMuted)
                        .multilineTextAlignment(.center).frame(maxWidth: 320)
                    Button("Try Again") { model.loadMesh() }.buttonStyle(GhostButtonStyle())
                }
            case .ready:
                overlays
            case .none:
                EmptyView()
            }
        }
        .clipped()
        .animation(.easeOut(duration: 0.15), value: showGrid)
        .fileExporter(isPresented: $model.showImageExporter, document: model.imageExport,
                      contentType: model.imageExport?.contentType ?? .png,
                      defaultFilename: model.imageExport?.name) { result in
            if case .failure(let error) = result { model.showToast(error.localizedDescription) }
            else { model.showToast(model.imageExport?.files.count == 1 ? "Image saved" : "Views saved", success: true) }
        }
    }

    private var overlays: some View {
        VStack(spacing: 0) {
            HStack(alignment: .top) {
                ViewerTools(showGrid: $showGrid, showDimensions: $showDimensions)
                Spacer()
                VStack(alignment: .trailing, spacing: 8) {
                    ViewPresetBar { model.viewCommand = ViewCommand(preset: $0) }
                    if showGrid {    // the compass belongs with the axes it names
                        AxisTriad(camera: camera) { model.viewCommand = ViewCommand(preset: $0) }
                            .transition(.opacity)
                    }
                }
            }
            Spacer()
            HStack(alignment: .bottom) {
                if let face = model.selectedFace {
                    SelectionCard(face: face, ask: model.askAboutSelection, close: model.clearSelection)
                } else if let shown = model.answerHighlight {
                    HighlightCard(highlight: shown, look: { model.viewCommand = ViewCommand(faces: shown.faces) },
                                  close: model.clearSelection)
                } else {
                    Text(hint).typo(10, .medium).foregroundStyle(p.textDim)
                }
                Spacer()
                if showGrid || showDimensions, let mesh = model.mesh {
                    ScaleLegend(mesh: mesh, grid: showGrid, dimensions: showDimensions)
                }
            }
        }
        .padding(10)
    }

    private var hint: String {
        let inspect = model.session?.isMesh == false
        #if os(macOS)
        return "Drag to rotate · Scroll to zoom · Option-drag to pan" + (inspect ? " · Click a face to inspect it" : "")
        #else
        return "Drag to rotate · Pinch to zoom · Two fingers to pan" + (inspect ? " · Tap a face to inspect it" : "")
        #endif
    }
}

/// Grid and Dimensions (each on its own), and Export — top left.
private struct ViewerTools: View {
    @Binding var showGrid: Bool
    @Binding var showDimensions: Bool
    @Environment(AppModel.self) private var model
    @Environment(\.palette) private var p

    var body: some View {
        HStack(spacing: 0) {
            ToolToggle(title: "Grid", icon: "grid", isOn: $showGrid,
                       help: "A measuring grid across the floor, with X, Y and Z axes marked in lengths")
            ToolToggle(title: "Dimensions", icon: "ruler", isOn: $showDimensions,
                       help: "The part's bounding box, with its overall length, width and height")

            Menu {
                Button("Save Current View as PNG…") { model.saveCurrentView() }
                Button("Export the 7 Standard Views…") { Task { await model.exportStandardViews() } }
                    .disabled(model.session?.images.isEmpty ?? true)
            } label: {
                Label("Export", systemImage: "square.and.arrow.up").labelStyle(.titleAndIcon)
                    .typo(10.5, .medium).foregroundStyle(p.textMuted)
            }
            .menuStyle(.button)
            .buttonStyle(.plain)
            .menuIndicator(.hidden)
            .fixedSize()
            .padding(.horizontal, 8).frame(height: 22)
            .help("Save images of the part")
        }
        .padding(.horizontal, 3)
        .background(p.surface.opacity(0.92), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).stroke(p.border, lineWidth: 1))
    }
}

private struct ToolToggle: View {
    var title: String
    var icon: String
    @Binding var isOn: Bool
    var help: String
    @Environment(\.palette) private var p

    var body: some View {
        Button { isOn.toggle() } label: {
            Label(title, systemImage: icon).labelStyle(.titleAndIcon)
                .typo(10.5, .medium)
                .foregroundStyle(isOn ? p.text : p.textMuted)
                .padding(.horizontal, 8).frame(height: 22)
                .background(isOn ? p.accentSoft : .clear, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(help)
    }
}

/// What the grid squares measure, and the part's overall size.
private struct ScaleLegend: View {
    var mesh: PartMesh
    var grid: Bool
    var dimensions: Bool
    @Environment(\.palette) private var p

    var body: some View {
        let s = mesh.size, u = mesh.units
        VStack(alignment: .trailing, spacing: 2) {
            if grid {
                Text("Squares \(PartMesh.format(mesh.gridStep)) \(u) · heavy lines every \(PartMesh.format(mesh.gridStep * 5)) \(u)")
            }
            if dimensions {
                Text("Overall \(PartMesh.format(s.x)) × \(PartMesh.format(s.y)) × \(PartMesh.format(s.z)) \(u)  (X × Y × Z)")
                    .foregroundStyle(grid ? p.textMuted : p.text)
            }
        }
        .typo(10.5, .medium).foregroundStyle(p.text)
        .padding(.horizontal, 10).padding(.vertical, 6)
        .background(p.surface.opacity(0.92), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).stroke(p.border, lineWidth: 1))
        .textSelection(.enabled)
    }
}

/// Which way the camera faces, shared from the renderer with the compass.
@Observable
final class CameraState {
    var orientation = simd_quatf(angle: 0, axis: SIMD3(0, 0, 1))
}

/// A small X/Y/Z compass that turns with the view; click an axis to look along it.
private struct AxisTriad: View {
    var camera: CameraState
    var choose: (ViewPreset) -> Void
    @Environment(\.palette) private var p

    private struct Axis: Identifiable {
        var name: String, direction: SIMD3<Float>, hex: UInt32, preset: ViewPreset
        var id: String { name }
    }

    var body: some View {
        let style = p.isDark ? ViewerStyle.dark : .light
        let axes = [Axis(name: "X", direction: SIMD3(1, 0, 0), hex: style.axisX, preset: .right),
                    Axis(name: "Y", direction: SIMD3(0, 1, 0), hex: style.axisY, preset: .back),
                    Axis(name: "Z", direction: SIMD3(0, 0, 1), hex: style.axisZ, preset: .top)]
        let toCamera = camera.orientation.inverse
        let size: CGFloat = 66, reach: CGFloat = 22
        // Far axes first, so the near ones draw over them.
        let placed = axes.map { a -> (Axis, CGPoint, Float) in
            let v = toCamera.act(a.direction)
            return (a, CGPoint(x: size / 2 + CGFloat(v.x) * reach, y: size / 2 - CGFloat(v.y) * reach), v.z)
        }.sorted { $0.2 < $1.2 }

        ZStack {
            Circle().fill(p.surface.opacity(0.9))
            Circle().stroke(p.border, lineWidth: 1)
            ForEach(placed, id: \.0.id) { axis, end, depth in
                let color = ViewerStyle.color(axis.hex)
                let away = depth < -0.2
                Path { $0.move(to: CGPoint(x: size / 2, y: size / 2)); $0.addLine(to: end) }
                    .stroke(color.opacity(away ? 0.45 : 1), style: StrokeStyle(lineWidth: 1.6, lineCap: .round))
                Button { choose(axis.preset) } label: {
                    Text(axis.name).font(.system(size: 8.5, weight: .bold)).foregroundStyle(.white)
                        .frame(width: 15, height: 15)
                        .background(Circle().fill(color.opacity(away ? 0.5 : 1)))
                }
                .buttonStyle(.plain)
                .position(end)
                .help("Look along \(axis.name) (\(axis.preset.label) view)")
            }
        }
        .frame(width: size, height: size)
    }
}

/// Iso · Front · Top · Right, with the rest under "More".
private struct ViewPresetBar: View {
    var choose: (ViewPreset) -> Void
    @Environment(\.palette) private var p

    var body: some View {
        HStack(spacing: 0) {
            ForEach([ViewPreset.iso, .front, .top, .right]) { preset in
                PresetButton(label: preset.label) { choose(preset) }
            }
            Menu {
                ForEach([ViewPreset.back, .left, .bottom]) { preset in
                    Button(preset.label) { choose(preset) }
                }
            } label: {
                Image(systemName: "ellipsis").font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(p.textMuted).frame(width: 26, height: 22).contentShape(Rectangle())
            }
            .menuStyle(.button)
            .buttonStyle(.plain)
            .menuIndicator(.hidden)
            .fixedSize()
            .help("More views")
        }
        .padding(.horizontal, 3)
        .background(p.surface.opacity(0.92), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).stroke(p.border, lineWidth: 1))
    }
}

private struct PresetButton: View {
    var label: String
    var action: () -> Void
    @Environment(\.palette) private var p
    @State private var hover = false

    var body: some View {
        Button(action: action) {
            Text(label).typo(10.5, .medium)
                .foregroundStyle(hover ? p.text : p.textMuted)
                .padding(.horizontal, 8).frame(height: 22).contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hover = $0 }
        .help("\(label) view")
    }
}

/// What the clicked face is: its feature first (that's what people mean), then the face.
private struct SelectionCard: View {
    var face: FaceInfo
    var ask: () -> Void
    var close: () -> Void
    @Environment(\.palette) private var p

    var body: some View {
        HStack(alignment: .center, spacing: 14) {
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(face.feature ?? face.kind).typo(12, .semibold).foregroundStyle(p.text)
                    if let detail = face.featureDetail ?? (face.feature == nil ? face.detail : nil) {
                        Text(detail).typo(12, .medium).foregroundStyle(p.text)
                    }
                }
                Text(subtitle).typo(10.5, .medium).foregroundStyle(p.textMuted)
            }
            .textSelection(.enabled)
            Button("Ask about this", action: ask).buttonStyle(PrimaryButtonStyle(compact: true))
            Button(action: close) {
                Image(systemName: "xmark").font(.system(size: 9, weight: .bold)).foregroundStyle(p.textMuted)
                    .frame(width: 18, height: 18).contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("Clear selection")
        }
        .padding(.leading, 12).padding(.trailing, 8).padding(.vertical, 8)
        .background(p.surface, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).stroke(p.border, lineWidth: 1))
        .shadow(color: .black.opacity(p.isDark ? 0.4 : 0.08), radius: 8, y: 2)
    }

    private var subtitle: String {
        var parts = [face.id]
        if face.feature != nil { parts.append([face.kind, face.detail].compactMap { $0 }.joined(separator: " ")) }
        if let c = face.component { parts.append("in \(c)") }
        return parts.joined(separator: " · ")
    }
}

/// What an answer pointed to in the view ("Thinnest wall · 1.6 mm"), until a click or × clears it.
private struct HighlightCard: View {
    var highlight: ViewerHighlight
    var look: () -> Void
    var close: () -> Void
    @Environment(\.palette) private var p

    var body: some View {
        HStack(alignment: .center, spacing: 14) {
            HStack(spacing: 9) {
                Image(systemName: "scope").font(.system(size: 12, weight: .semibold)).foregroundStyle(p.accent)
                VStack(alignment: .leading, spacing: 3) {
                    Text(highlight.label.isEmpty ? "From the answer" : highlight.label)
                        .typo(12, .semibold).foregroundStyle(p.text)
                    Text(subtitle).typo(10.5, .medium).foregroundStyle(p.textMuted)
                }
            }
            .textSelection(.enabled)
            Button("Look at it", action: look).buttonStyle(GhostButtonStyle())
                .help("Turn the view back to the highlighted faces")
            Button(action: close) {
                Image(systemName: "xmark").font(.system(size: 9, weight: .bold)).foregroundStyle(p.textMuted)
                    .frame(width: 18, height: 18).contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("Clear the highlight")
        }
        .padding(.leading, 12).padding(.trailing, 8).padding(.vertical, 8)
        .background(p.surface, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).stroke(p.border, lineWidth: 1))
        .shadow(color: .black.opacity(p.isDark ? 0.4 : 0.08), radius: 8, y: 2)
    }

    private var subtitle: String {
        let faces = highlight.faces.count == 1 ? "F\(highlight.faces[0])" : "\(highlight.faces.count) faces"
        return "From the answer · \(faces)"
    }
}

/// The draggable line between the 3D view and the chat. It sets the view's *share* of the
/// window, so the split keeps its proportions as the window grows or shrinks.
struct ViewerDivider: View {
    @Binding var fraction: Double
    var total: Double
    var range: ClosedRange<Double>
    @Environment(\.palette) private var p
    @State private var start: Double?

    var body: some View {
        ZStack {
            p.surface
            Capsule().fill(p.border).frame(width: 30, height: 3)
        }
        .frame(height: 8)
        .overlay(alignment: .top) { Hairline() }
        .overlay(alignment: .bottom) { Hairline() }
        .contentShape(Rectangle())
        .gesture(DragGesture(minimumDistance: 1, coordinateSpace: .global)
            .onChanged { v in
                if start == nil { start = min(max(fraction * total, range.lowerBound), range.upperBound) }
                let height = min(max(start! + v.translation.height, range.lowerBound), range.upperBound)
                fraction = total > 0 ? height / total : fraction
            }
            .onEnded { _ in start = nil })
        #if os(macOS)
        .onHover { inside in if inside { NSCursor.resizeUpDown.push() } else { NSCursor.pop() } }
        #endif
    }
}

// MARK: - SceneKit

private struct ViewerStyle {
    var background: UInt32
    var body: UInt32
    var edge: UInt32, edgeAlpha: Double
    var highlight: UInt32
    var gridMinor: UInt32, gridMajor: UInt32
    var axisX: UInt32, axisY: UInt32, axisZ: UInt32
    var tick: UInt32
    var box: UInt32, boxAlpha: Double
    var label: UInt32
    var ambient: CGFloat

    static let light = ViewerStyle(background: 0xF1F1EE, body: 0xB9C2CC, edge: 0x2B323B, edgeAlpha: 0.85,
                                   highlight: 0x2A78D6, gridMinor: 0x9C9C96, gridMajor: 0x74746E,
                                   axisX: 0xD13438, axisY: 0x1F9D55, axisZ: 0x2F6BE8, tick: 0x6B6B6B,
                                   box: 0x2E2E2E, boxAlpha: 0.75, label: 0x2E2E2E, ambient: 420)
    static let dark = ViewerStyle(background: 0x141419, body: 0x8C97A5, edge: 0x0A0C0F, edgeAlpha: 0.9,
                                  highlight: 0x5B6CFF, gridMinor: 0x4A4D58, gridMajor: 0x6B6F7D,
                                  axisX: 0xE5484D, axisY: 0x3DB37A, axisZ: 0x4C82F7, tick: 0x8A8A93,
                                  box: 0xD0D0D6, boxAlpha: 0.7, label: 0xD0D0D6, ambient: 330)

    static func color(_ hex: UInt32, _ alpha: Double = 1) -> Color { Color(hex: hex, alpha: alpha) }

    /// A colour as SceneKit wants it in a vertex buffer: linear light, not sRGB.
    static func linear(_ hex: UInt32) -> SIMD3<Float> {
        func channel(_ v: UInt32) -> Float {
            let c = Float(v & 0xFF) / 255
            return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4)
        }
        return SIMD3(channel(hex >> 16), channel(hex >> 8), channel(hex))
    }
}

#if os(macOS)
private typealias PlatformFontWeight = NSFont.Weight
#else
private typealias PlatformFontWeight = UIFont.Weight
#endif

private extension Color {
    #if os(macOS)
    var platform: NSColor { NSColor(self) }
    #else
    var platform: UIColor { UIColor(self) }
    #endif
}

@MainActor
struct ModelViewer {
    var mesh: PartMesh?
    var highlighted: [Int]
    var command: ViewCommand
    var dark: Bool
    var showGrid: Bool
    var showDimensions: Bool
    var snapshotRequest: UUID?
    var camera: CameraState
    var onSnapshot: (Data?) -> Void
    var onPick: (Int?) -> Void

    func makeCoordinator() -> ViewerController { ViewerController() }

    fileprivate func configure(_ view: SCNView, _ c: ViewerController) {
        view.scene = c.scene
        view.pointOfView = c.cameraNode
        view.allowsCameraControl = true
        view.antialiasingMode = .multisampling4X
        view.preferredFramesPerSecond = 60
        let control = view.defaultCameraController
        control.interactionMode = .orbitTurntable
        control.inertiaEnabled = true
        control.worldUp = SCNVector3(0, 0, 1)
        view.cameraControlConfiguration.autoSwitchToFreeCamera = false
        view.cameraControlConfiguration.allowsTranslation = true
        view.delegate = c.labelScaler
        c.labelScaler.onOrientation = { [camera] in camera.orientation = $0 }
        c.view = view
    }

    fileprivate func update(_ c: ViewerController) {
        c.onPick = onPick
        c.setDark(dark)
        c.show(mesh)
        c.setOverlays(grid: showGrid, dimensions: showDimensions)
        c.setHighlight(highlighted)
        c.apply(command)
        if let snapshotRequest, c.takeSnapshot(snapshotRequest) {
            let png = c.snapshotPNG()
            DispatchQueue.main.async { onSnapshot(png) }
        }
    }
}

#if os(macOS)
extension ModelViewer: NSViewRepresentable {
    func makeNSView(context: Context) -> PickingSCNView {
        let view = PickingSCNView(frame: .zero, options: nil)
        configure(view, context.coordinator)
        view.onClick = { [weak c = context.coordinator] in c?.pick(at: $0) }
        view.onDoubleClick = { [weak c = context.coordinator] in c?.focus(at: $0) }
        view.onResize = { [scaler = context.coordinator.labelScaler] in scaler.setViewportHeight(Float($0.height)) }
        return view
    }
    func updateNSView(_ view: PickingSCNView, context: Context) { update(context.coordinator) }
}

/// SCNView's camera control owns the mouse; a press that barely moves is a click.
final class PickingSCNView: SCNView {
    var onResize: (CGSize) -> Void = { _ in }
    var onClick: (CGPoint) -> Void = { _ in }
    var onDoubleClick: (CGPoint) -> Void = { _ in }
    private var downAt: CGPoint?

    override func setFrameSize(_ newSize: NSSize) {
        super.setFrameSize(newSize)
        onResize(newSize)
    }

    override func mouseDown(with event: NSEvent) {
        downAt = convert(event.locationInWindow, from: nil)
        super.mouseDown(with: event)
    }

    override func mouseUp(with event: NSEvent) {
        super.mouseUp(with: event)
        let at = convert(event.locationInWindow, from: nil)
        if let d = downAt, hypot(at.x - d.x, at.y - d.y) < 4 {
            event.clickCount >= 2 ? onDoubleClick(at) : onClick(at)
        }
        downAt = nil
    }
}
#else
extension ModelViewer: UIViewRepresentable {
    func makeUIView(context: Context) -> SCNView {
        let view = SCNView(frame: .zero, options: nil)
        configure(view, context.coordinator)
        let c = context.coordinator
        let doubleTap = UITapGestureRecognizer(target: c, action: #selector(ViewerController.doubleTapped(_:)))
        doubleTap.numberOfTapsRequired = 2
        let tap = UITapGestureRecognizer(target: c, action: #selector(ViewerController.tapped(_:)))
        tap.require(toFail: doubleTap)
        view.addGestureRecognizer(doubleTap)
        view.addGestureRecognizer(tap)
        return view
    }
    func updateUIView(_ view: SCNView, context: Context) { update(context.coordinator) }
}
#endif

/// Owns the scene: the part, its edges, the highlight, the camera and the lights.
@MainActor
final class ViewerController: NSObject {
    let scene = SCNScene()
    let cameraNode = SCNNode()
    weak var view: SCNView?
    var onPick: (Int?) -> Void = { _ in }

    private var mesh: PartMesh?
    private var partNode: SCNNode?
    private var highlightNode: SCNNode?, ghostNode: SCNNode?
    private var sources: [SCNGeometrySource] = []
    private var highlighted: [Int] = []
    private var lastCommand: UUID?
    private var lastSnapshot: UUID?
    private var dark: Bool?
    private var style = ViewerStyle.dark
    private var gridOn = false, dimensionsOn = false
    private var gridNode: SCNNode?, dimensionsNode: SCNNode?
    private let ambient = SCNLight()
    private let bodyMaterial = SCNMaterial()
    private let edgeMaterial = SCNMaterial()
    private let highlightMaterial = SCNMaterial()
    private let ghostMaterial = SCNMaterial()
    private let gridMaterial = SCNMaterial()
    private let boxMaterial = SCNMaterial()
    private let labelMaterial = SCNMaterial()
    private let tickMaterial = SCNMaterial()

    private static let partMask = 1, overlayMask = 2
    let labelScaler = LabelScaler()

    override init() {
        super.init()
        let camera = SCNCamera()
        camera.fieldOfView = 30
        camera.automaticallyAdjustsZRange = true
        cameraNode.camera = camera
        scene.rootNode.addChildNode(cameraNode)

        // A key light riding on the camera, from the upper left, so whatever you look at is
        // lit; a weak fill from the lower right; and ambient so nothing goes black.
        for (intensity, angles) in [(780.0, SCNVector3(-0.35, -0.3, 0)), (170.0, SCNVector3(0.5, 0.6, 0))] {
            let light = SCNLight()
            light.type = .directional
            light.intensity = intensity
            let node = SCNNode()
            node.light = light
            node.eulerAngles = angles
            cameraNode.addChildNode(node)
        }
        ambient.type = .ambient
        let ambientNode = SCNNode()
        ambientNode.light = ambient
        scene.rootNode.addChildNode(ambientNode)

        bodyMaterial.lightingModel = .blinn
        bodyMaterial.specular.contents = Color.white.platform
        bodyMaterial.specular.intensity = 0.16
        bodyMaterial.isDoubleSided = true     // open meshes and inside faces still draw

        edgeMaterial.lightingModel = .constant
        edgeMaterial.shaderModifiers = [.geometry: Self.pullTowardCamera(0.9994)]

        highlightMaterial.lightingModel = .blinn
        highlightMaterial.isDoubleSided = true
        highlightMaterial.shaderModifiers = [.geometry: Self.pullTowardCamera(0.9997)]
        // The same faces again, faintly, drawn over everything: a highlight inside a pocket or
        // round the back still shows where it is.
        ghostMaterial.lightingModel = .constant
        ghostMaterial.isDoubleSided = true
        ghostMaterial.readsFromDepthBuffer = false
        ghostMaterial.writesToDepthBuffer = false
        ghostMaterial.transparency = 0.22

        gridMaterial.lightingModel = .constant
        gridMaterial.isDoubleSided = true
        gridMaterial.writesToDepthBuffer = false
        gridMaterial.blendMode = .alpha
        gridMaterial.shaderModifiers = [.fragment: Self.gridShader]
        boxMaterial.lightingModel = .constant
        labelMaterial.lightingModel = .constant
        labelMaterial.isDoubleSided = true
        labelMaterial.readsFromDepthBuffer = false   // overall sizes stay readable even behind the part
        tickMaterial.lightingModel = .constant
        tickMaterial.isDoubleSided = true            // axis markings hide behind the part, like the axes
    }

    /// The floor grid, per pixel. Line distance is measured in screen pixels (via fwidth), so
    /// lines are anti-aliased and one pixel wide at any range; each set of lines fades out as
    /// its squares shrink toward a few pixels; everything fades with distance from the part.
    private static let gridShader = """
    #pragma arguments
    float cvMinorStep;
    float cvMajorStep;
    float cvFadeRadius;
    float3 cvFadeCenter;
    float3 cvAxisPoint;
    float3 cvMinorColor;
    float3 cvMajorColor;
    float3 cvXAxisColor;
    float3 cvYAxisColor;
    #pragma transparent
    #pragma body
    float4 world = scn_frame.inverseViewTransform * float4(_surface.position, 1.0);
    float2 p = world.xy;
    float2 w = max(fwidth(p), float2(1e-6));
    float2 gm = abs(fract(p / cvMinorStep - 0.5) - 0.5) * cvMinorStep / w;
    float2 gM = abs(fract(p / cvMajorStep - 0.5) - 0.5) * cvMajorStep / w;
    float minorLine = 1.0 - min(min(gm.x, gm.y), 1.0);
    float majorLine = 1.0 - min(min(gM.x, gM.y) / 1.25, 1.0);
    float cell = max(w.x, w.y);
    float minorVisible = 1.0 - smoothstep(0.10, 0.22, cell / cvMinorStep);
    float majorVisible = 1.0 - smoothstep(0.10, 0.22, cell / cvMajorStep);
    float xAxis = 1.0 - min(abs(p.y - cvAxisPoint.y) / w.y / 1.4, 1.0);
    float yAxis = 1.0 - min(abs(p.x - cvAxisPoint.x) / w.x / 1.4, 1.0);
    float fade = 1.0 - smoothstep(cvFadeRadius * 0.25, cvFadeRadius, length(p - cvFadeCenter.xy));
    float3 color = cvMinorColor;
    float alpha = minorLine * minorVisible * 0.5;
    float majorAlpha = majorLine * majorVisible * 0.85;
    if (majorAlpha > alpha) { color = cvMajorColor; alpha = majorAlpha; }
    if (xAxis > 0.0) { color = mix(color, cvXAxisColor, xAxis); alpha = max(alpha, xAxis * 0.9); }
    if (yAxis > 0.0) { color = mix(color, cvYAxisColor, yAxis); alpha = max(alpha, yAxis * 0.9); }
    alpha *= fade;
    // Seen from below, the floor is between you and the part: let it recede.
    float4 eye = scn_frame.inverseViewTransform * float4(0.0, 0.0, 0.0, 1.0);
    if (eye.z < world.z) { alpha *= 0.3; }
    _output.color = float4(color * alpha, alpha);
    """

    /// Edges and highlights sit exactly on the faces; nudging them toward the eye (by a
    /// fraction of the distance, so it holds at any zoom) keeps them from flickering.
    private static func pullTowardCamera(_ factor: Double) -> String {
        """
        float4 eye = scn_node.modelViewTransform * _geometry.position;
        eye.xyz *= \(factor);
        _geometry.position = scn_node.inverseModelViewTransform * eye;
        """
    }

    func setDark(_ dark: Bool) {
        guard dark != self.dark else { return }
        self.dark = dark
        style = dark ? ViewerStyle.dark : .light
        scene.background.contents = ViewerStyle.color(style.background).platform
        bodyMaterial.diffuse.contents = ViewerStyle.color(style.body).platform
        edgeMaterial.diffuse.contents = ViewerStyle.color(style.edge, style.edgeAlpha).platform
        highlightMaterial.diffuse.contents = ViewerStyle.color(style.highlight).platform
        ghostMaterial.diffuse.contents = ViewerStyle.color(style.highlight).platform
        boxMaterial.diffuse.contents = ViewerStyle.color(style.box, style.boxAlpha).platform
        labelMaterial.diffuse.contents = ViewerStyle.color(style.label).platform
        tickMaterial.diffuse.contents = ViewerStyle.color(style.tick).platform
        ambient.intensity = style.ambient
        if gridOn { buildGrid() }      // the grid's colours are baked into its vertices
    }

    func show(_ mesh: PartMesh?) {
        guard mesh?.sessionId != self.mesh?.sessionId else { return }
        partNode?.removeFromParentNode()
        partNode = nil
        highlightNode = nil
        ghostNode = nil
        highlighted = []
        gridNode?.removeFromParentNode(); gridNode = nil
        dimensionsNode?.removeFromParentNode(); dimensionsNode = nil
        refreshLabels()
        self.mesh = mesh
        guard let mesh else { return }

        let vertices = SCNGeometrySource(data: mesh.positions, semantic: .vertex, vectorCount: mesh.vertexCount,
                                         usesFloatComponents: true, componentsPerVector: 3, bytesPerComponent: 4,
                                         dataOffset: 0, dataStride: 12)
        let normals = SCNGeometrySource(data: mesh.normals, semantic: .normal, vectorCount: mesh.vertexCount,
                                        usesFloatComponents: true, componentsPerVector: 3, bytesPerComponent: 4,
                                        dataOffset: 0, dataStride: 12)
        sources = [vertices, normals]
        let triangles = SCNGeometryElement(data: mesh.indices, primitiveType: .triangles,
                                           primitiveCount: mesh.triangleCount, bytesPerIndex: 4)
        let body = SCNGeometry(sources: sources, elements: [triangles])
        body.materials = [bodyMaterial]
        let node = SCNNode(geometry: body)
        node.categoryBitMask = Self.partMask

        if mesh.edgeSegmentCount > 0 {
            let points = SCNGeometrySource(data: mesh.edgePoints, semantic: .vertex, vectorCount: mesh.edgePointCount,
                                           usesFloatComponents: true, componentsPerVector: 3, bytesPerComponent: 4,
                                           dataOffset: 0, dataStride: 12)
            let lines = SCNGeometryElement(data: mesh.edgeSegments, primitiveType: .line,
                                           primitiveCount: mesh.edgeSegmentCount, bytesPerIndex: 4)
            let edges = SCNGeometry(sources: [points], elements: [lines])
            edges.materials = [edgeMaterial]
            let edgeNode = SCNNode(geometry: edges)
            edgeNode.categoryBitMask = Self.overlayMask
            edgeNode.renderingOrder = 1
            node.addChildNode(edgeNode)
        }
        scene.rootNode.addChildNode(node)
        partNode = node
        if gridOn { buildGrid() }
        if dimensionsOn { buildDimensions() }
    }

    // MARK: Grid and dimensions

    func setOverlays(grid: Bool, dimensions: Bool) {
        if grid != gridOn {
            gridOn = grid
            if grid { buildGrid() } else { gridNode?.removeFromParentNode(); gridNode = nil; refreshLabels() }
        }
        if dimensions != dimensionsOn {
            dimensionsOn = dimensions
            if dimensions { buildDimensions() } else { dimensionsNode?.removeFromParentNode(); dimensionsNode = nil; refreshLabels() }
        }
    }

    /// A floor under the part reaching far past it, drawn per pixel on the GPU: every line
    /// stays one crisp pixel at any distance, fine lines fade out once they'd crowd
    /// together, and the whole floor dissolves toward the horizon. The X (red) and Y (green)
    /// axes run through the model's origin when it's on the floor, else the part's corner;
    /// Z (blue) stands up from there. Lengths are marked along all three near the part.
    private func buildGrid() {
        gridNode?.removeFromParentNode()
        gridNode = nil
        guard let mesh else { return }
        let lo = mesh.boundsMin, hi = mesh.boundsMax, mid = (lo + hi) / 2
        let step = mesh.gridStep, major = step * 5
        let z = lo.z - mesh.radius * 0.002        // a hair under the part, so its base doesn't flicker
        let reach = max(mesh.radius * 10, major * 6)
        let near = mesh.radius * 2.6              // lengths are marked out to here
        let originOnFloor = simd_length(SIMD2<Float>(mid.x, mid.y)) < reach * 0.5
        let axisAt = originOnFloor ? SIMD2<Float>(0, 0) : SIMD2(lo.x, lo.y)

        let floor = SCNPlane(width: CGFloat(reach * 2), height: CGFloat(reach * 2))
        floor.materials = [gridMaterial]
        let floorNode = SCNNode(geometry: floor)
        floorNode.simdPosition = SIMD3(mid.x, mid.y, z)
        floorNode.categoryBitMask = Self.overlayMask
        floorNode.renderingOrder = 5              // after the part, so the part hides what's under it
        func vector(_ v: SIMD3<Float>) -> NSValue { NSValue(scnVector3: SCNVector3(v)) }
        gridMaterial.setValue(NSNumber(value: step), forKey: "cvMinorStep")
        gridMaterial.setValue(NSNumber(value: major), forKey: "cvMajorStep")
        gridMaterial.setValue(NSNumber(value: reach), forKey: "cvFadeRadius")
        gridMaterial.setValue(vector(SIMD3(mid.x, mid.y, 0)), forKey: "cvFadeCenter")
        gridMaterial.setValue(vector(SIMD3(axisAt.x, axisAt.y, 0)), forKey: "cvAxisPoint")
        gridMaterial.setValue(vector(ViewerStyle.linear(style.gridMinor)), forKey: "cvMinorColor")
        gridMaterial.setValue(vector(ViewerStyle.linear(style.gridMajor)), forKey: "cvMajorColor")
        gridMaterial.setValue(vector(ViewerStyle.linear(style.axisX)), forKey: "cvXAxisColor")
        gridMaterial.setValue(vector(ViewerStyle.linear(style.axisY)), forKey: "cvYAxisColor")

        let root = SCNNode()
        root.addChildNode(floorNode)
        let top = max(hi.z, z + major) + step
        let zMaterial = SCNMaterial()
        zMaterial.lightingModel = .constant
        zMaterial.diffuse.contents = ViewerStyle.color(style.axisZ).platform
        root.addChildNode(Self.lines([SIMD3(axisAt.x, axisAt.y, z), SIMD3(axisAt.x, axisAt.y, top)], material: zMaterial))

        // Lengths at the heavy lines near the part, in the model's own coordinates.
        let offset = step * 0.9
        let center = SIMD2(mid.x, mid.y)
        for m in stride(from: (axisAt.x - near) / major, through: (axisAt.x + near) / major, by: 1).map({ $0.rounded() * major })
        where simd_distance(SIMD2(m, axisAt.y), center) < near {
            root.addChildNode(label(PartMesh.format(m, signed: true), at: SIMD3(m, axisAt.y - offset, z),
                                    points: 9.5, material: tickMaterial))
        }
        for m in stride(from: (axisAt.y - near) / major, through: (axisAt.y + near) / major, by: 1).map({ $0.rounded() * major })
        where simd_distance(SIMD2(axisAt.x, m), center) < near && abs(m - axisAt.y) > step * 0.01 {
            root.addChildNode(label(PartMesh.format(m, signed: true), at: SIMD3(axisAt.x - offset, m, z),
                                    points: 9.5, material: tickMaterial))
        }
        var zm = (z / major).rounded(.up) * major
        while zm <= top - step * 0.5 {
            if abs(zm - z) > step * 0.5 {
                root.addChildNode(label(PartMesh.format(zm, signed: true),
                                        at: SIMD3(axisAt.x - offset * 0.7, axisAt.y - offset * 0.7, zm),
                                        points: 9.5, material: tickMaterial))
            }
            zm += major
        }
        // Axis names, in the axis colours, just past the last marked length.
        for (name, hex, at) in [("X", style.axisX, SIMD3(axisAt.x + near + major * 0.5, axisAt.y - offset, z)),
                                ("Y", style.axisY, SIMD3(axisAt.x - offset, axisAt.y + near + major * 0.5, z)),
                                ("Z", style.axisZ, SIMD3(axisAt.x, axisAt.y, top + offset))] {
            let material = SCNMaterial()
            material.lightingModel = .constant
            material.isDoubleSided = true
            material.diffuse.contents = ViewerStyle.color(hex).platform
            root.addChildNode(label(name, at: at, points: 11, material: material, weight: .bold))
        }
        scene.rootNode.addChildNode(root)
        gridNode = root
        refreshLabels()
    }

    /// Overall length, width and height as an engineering drawing shows them: dimension lines
    /// standing off the part on extension lines, with arrowheads and the size at the middle.
    private func buildDimensions() {
        dimensionsNode?.removeFromParentNode()
        dimensionsNode = nil
        guard let mesh else { return }
        let lo = mesh.boundsMin, hi = mesh.boundsMax, size = mesh.size
        // How far the dimension line stands off: clear of the grid's axis numbers too.
        let gap = max(mesh.radius * 0.16, mesh.gridStep * 2.2)
        var lines: [SIMD3<Float>] = []
        let root = SCNNode()

        /// One dimension from *a* to *b*, standing off along *out* (unit), labelled *text*.
        func dimension(_ a: SIMD3<Float>, _ b: SIMD3<Float>, out: SIMD3<Float>, text: String) {
            let a1 = a + out * gap, b1 = b + out * gap
            let start = out * gap * 0.15, past = out * gap * 1.18
            lines += [a + start, a + past, b + start, b + past]          // extension lines
            lines += [a1, b1]                                            // dimension line
            let along = simd_normalize(b - a), length = simd_distance(a, b)
            let head = min(gap * 0.32, length * 0.12), spread = head * 0.32
            for (tip, dir) in [(a1, along), (b1, -along)] {             // open arrowheads
                lines += [tip, tip + dir * head + out * spread, tip, tip + dir * head - out * spread]
            }
            root.addChildNode(label(text, at: (a1 + b1) / 2 + out * gap * 0.38, points: 11.5,
                                    material: labelMaterial, weight: .semibold))
        }
        let u = mesh.units
        if size.x > 0 { dimension(SIMD3(lo.x, lo.y, lo.z), SIMD3(hi.x, lo.y, lo.z), out: SIMD3(0, -1, 0),
                                  text: "\(PartMesh.format(size.x)) \(u)") }
        if size.y > 0 { dimension(SIMD3(hi.x, lo.y, lo.z), SIMD3(hi.x, hi.y, lo.z), out: SIMD3(1, 0, 0),
                                  text: "\(PartMesh.format(size.y)) \(u)") }
        if size.z > 0 { dimension(SIMD3(hi.x, lo.y, lo.z), SIMD3(hi.x, lo.y, hi.z),
                                  out: simd_normalize(SIMD3(1, -1, 0)), text: "\(PartMesh.format(size.z)) \(u)") }
        root.addChildNode(Self.lines(lines, material: boxMaterial))
        scene.rootNode.addChildNode(root)
        dimensionsNode = root
        refreshLabels()
    }

    private static func lines(_ points: [SIMD3<Float>], material: SCNMaterial) -> SCNNode {
        let source = SCNGeometrySource(vertices: points.map { SCNVector3($0) })
        let element = SCNGeometryElement(indices: (0 ..< UInt32(points.count)).map { $0 },
                                         primitiveType: .line)
        let geometry = SCNGeometry(sources: [source], elements: [element])
        geometry.materials = [material]
        let node = SCNNode(geometry: geometry)
        node.categoryBitMask = overlayMask
        return node
    }

    /// Text that always faces the camera and stays *points* tall on screen at any distance.
    private func label(_ text: String, at point: SIMD3<Float>, points: Float, material: SCNMaterial,
                       weight: PlatformFontWeight = .medium) -> SCNNode {
        let fontSize: CGFloat = 12
        let geometry = SCNText(string: text, extrusionDepth: 0)
        geometry.font = .monospacedDigitSystemFont(ofSize: fontSize, weight: weight)
        geometry.flatness = 0.02
        geometry.materials = [material]
        let text = SCNNode(geometry: geometry)
        let (a, b) = text.boundingBox
        text.pivot = SCNMatrix4MakeTranslation((a.x + b.x) / 2, (a.y + b.y) / 2, 0)
        text.simdScale = SIMD3(repeating: 1 / Float(fontSize))      // one unit per em; LabelScaler sizes it
        let holder = ScreenLabelNode(points: points)
        holder.simdPosition = point
        holder.constraints = [SCNBillboardConstraint()]
        holder.renderingOrder = 10
        text.renderingOrder = 10
        holder.categoryBitMask = Self.overlayMask
        holder.addChildNode(text)
        return holder
    }

    private func refreshLabels() {
        let all = [gridNode, dimensionsNode].compactMap { $0 }.flatMap(\.childNodes).compactMap { $0 as? ScreenLabelNode }
        labelScaler.set(all)
    }

    // MARK: Snapshot

    /// True once per request token.
    func takeSnapshot(_ token: UUID) -> Bool {
        guard token != lastSnapshot else { return false }
        lastSnapshot = token
        return true
    }

    /// What the view shows right now, rendered offscreen at twice its size.
    func snapshotPNG() -> Data? {
        guard let view, view.bounds.width > 0, view.bounds.height > 0 else { return nil }
        let renderer = SCNRenderer(device: view.device, options: nil)
        renderer.scene = scene
        renderer.pointOfView = cameraNode
        let size = CGSize(width: view.bounds.width * 2, height: view.bounds.height * 2)
        let image = renderer.snapshot(atTime: 0, with: size, antialiasingMode: .multisampling4X)
        #if os(macOS)
        guard let tiff = image.tiffRepresentation, let rep = NSBitmapImageRep(data: tiff) else { return nil }
        return rep.representation(using: .png, properties: [:])
        #else
        return image.pngData()
        #endif
    }

    func setHighlight(_ faces: [Int]) {
        guard faces != highlighted else { return }
        highlighted = faces
        for node in [highlightNode, ghostNode] { node?.removeFromParentNode() }
        highlightNode = nil
        ghostNode = nil
        guard let mesh, let partNode, !faces.isEmpty else { return }
        let (data, count) = mesh.indices(ofFaces: faces)
        guard count > 0 else { return }
        let element = SCNGeometryElement(data: data, primitiveType: .triangles, primitiveCount: count, bytesPerIndex: 4)
        func overlay(_ material: SCNMaterial, order: Int) -> SCNNode {
            let geometry = SCNGeometry(sources: sources, elements: [element])
            geometry.materials = [material]
            let node = SCNNode(geometry: geometry)
            node.categoryBitMask = Self.overlayMask
            node.renderingOrder = order
            partNode.addChildNode(node)
            return node
        }
        highlightNode = overlay(highlightMaterial, order: 2)
        ghostNode = overlay(ghostMaterial, order: 4)      // before the floor (5) and labels (10)
    }

    /// Frame the whole part from a standard direction, or turn to some faces.
    func apply(_ command: ViewCommand) {
        guard command.token != lastCommand, let mesh, view != nil else { return }
        lastCommand = command.token
        switch command.target {
        case .preset(let preset):
            look(at: mesh.center, radius: mesh.radius, from: preset.direction, up: preset.up, animated: command.animated)
        case .faces(let faces):
            guard let region = mesh.region(ofFaces: faces) else { return }
            // Keep the angle the user chose unless the faces turn away from it; then swing round
            // toward them. Come close enough to see them, never so close the part is lost.
            let target = view.map { SIMD3<Float>($0.defaultCameraController.target) } ?? mesh.center
            var direction = simd_normalize(cameraNode.simdPosition - target)
            if simd_length(region.facing) > 0.5 {
                let facing = simd_normalize(region.facing)
                if simd_dot(direction, facing) < 0.2 { direction = simd_normalize(facing * 1.5 + direction) }
            }
            let radius = min(max(region.radius * 2.5, mesh.radius * 0.3), mesh.radius)
            look(at: region.center, radius: radius, from: direction, up: SIMD3(0, 0, 1), animated: true)
        }
    }

    /// Put the camera on *direction* from *center*, just far enough back to fit a sphere of *radius*.
    private func look(at center: SIMD3<Float>, radius: Float, from direction: SIMD3<Float>, up: SIMD3<Float>,
                      animated: Bool) {
        guard let view else { return }
        let size = view.bounds.size
        let aspect = size.height > 0 ? Float(size.width / size.height) : 1.4
        let halfVertical = Float(cameraNode.camera?.fieldOfView ?? 30) * .pi / 360
        let halfHorizontal = atan(tan(halfVertical) * aspect)
        let distance = radius / sin(min(halfVertical, halfHorizontal)) * 1.04

        SCNTransaction.begin()
        SCNTransaction.animationDuration = animated ? 0.45 : 0
        SCNTransaction.animationTimingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
        let eye = center + direction * distance
        cameraNode.simdPosition = eye
        cameraNode.simdOrientation = Self.orientation(from: eye, to: center, up: up)
        SCNTransaction.commit()
        view.defaultCameraController.target = SCNVector3(center)
    }

    /// A camera orientation looking from *eye* to *target* with *up* as near to up as it can be.
    /// (SceneKit's own look-at turns the camera away from the part for the bottom view.)
    static func orientation(from eye: SIMD3<Float>, to target: SIMD3<Float>, up: SIMD3<Float>) -> simd_quatf {
        let forward = simd_normalize(target - eye)
        var right = simd_cross(forward, up)
        if simd_length(right) < 1e-5 {
            right = simd_cross(forward, abs(forward.z) < 0.9 ? SIMD3(0, 0, 1) : SIMD3(0, 1, 0))
        }
        right = simd_normalize(right)
        return simd_quatf(simd_float3x3(columns: (right, simd_cross(right, forward), -forward)))
    }

    func pick(at point: CGPoint) {
        onPick(hit(at: point).flatMap { mesh?.face(ofTriangle: $0.faceIndex) })
    }

    /// Double-click: turn to face that point and orbit around it from now on.
    func focus(at point: CGPoint) {
        guard let hit = hit(at: point), let view else { return }
        let target = hit.simdWorldCoordinates
        SCNTransaction.begin()
        SCNTransaction.animationDuration = 0.35
        SCNTransaction.animationTimingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
        cameraNode.simdOrientation = Self.orientation(from: cameraNode.simdPosition, to: target, up: cameraNode.simdWorldUp)
        SCNTransaction.commit()
        view.defaultCameraController.target = SCNVector3(target)
    }

    private func hit(at point: CGPoint) -> SCNHitTestResult? {
        guard let view, let partNode else { return nil }
        return view.hitTest(point, options: [
            .searchMode: SCNHitTestSearchMode.closest.rawValue,
            .categoryBitMask: Self.partMask,
            .backFaceCulling: false,
        ]).first { $0.node === partNode }
    }

    #if os(iOS)
    @objc func tapped(_ g: UITapGestureRecognizer) {
        guard let view = g.view else { return }
        pick(at: g.location(in: view))
    }
    @objc func doubleTapped(_ g: UITapGestureRecognizer) {
        guard let view = g.view else { return }
        focus(at: g.location(in: view))
    }
    #endif
}

/// A label that keeps the same size on screen; `points` is its height in points.
final class ScreenLabelNode: SCNNode {
    let points: Float
    init(points: Float) { self.points = points; super.init() }
    required init?(coder: NSCoder) { fatalError("not used") }
}

/// Rescales labels every frame so they stay readable whether the camera is near or far:
/// world height = points × (world per point at that distance).
final class LabelScaler: NSObject, SCNSceneRendererDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private var labels: [ScreenLabelNode] = []
    private var viewportHeight: Float = 400
    private var lastOrientation: simd_quatf?
    /// Called on the main thread when the camera turns (drives the compass).
    var onOrientation: @MainActor (simd_quatf) -> Void = { _ in }

    func set(_ labels: [ScreenLabelNode]) { lock.lock(); self.labels = labels; lock.unlock() }
    func setViewportHeight(_ h: Float) { lock.lock(); viewportHeight = max(h, 1); lock.unlock() }

    func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
        guard let eye = renderer.pointOfView, let camera = eye.camera else { return }
        let turned = eye.simdWorldOrientation
        lock.lock()
        let labels = self.labels, height = viewportHeight
        let changed = lastOrientation.map { abs(simd_dot($0.vector, turned.vector)) < 0.999_999 } ?? true
        if changed { lastOrientation = turned }
        let report = onOrientation
        lock.unlock()
        if changed { DispatchQueue.main.async { report(turned) } }
        guard !labels.isEmpty else { return }
        let perPoint = 2 * tan(Float(camera.fieldOfView) * .pi / 360) / height
        let at = eye.simdWorldPosition
        for label in labels {
            label.simdScale = SIMD3(repeating: label.points * perPoint * simd_distance(label.simdWorldPosition, at))
        }
    }
}

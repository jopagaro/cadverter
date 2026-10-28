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

    var body: some View {
        ZStack {
            ModelViewer(mesh: model.mesh, highlighted: model.highlightedFaces,
                        command: model.viewCommand, dark: p.isDark) { model.pickFace($0) }

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
    }

    private var overlays: some View {
        VStack(spacing: 0) {
            HStack {
                Spacer()
                ViewPresetBar { model.viewCommand = ViewCommand(preset: $0) }
            }
            Spacer()
            HStack(alignment: .bottom) {
                if let face = model.selectedFace {
                    SelectionCard(face: face, ask: model.askAboutSelection, close: model.clearSelection)
                } else {
                    Text(hint).typo(10, .medium).foregroundStyle(p.textDim)
                }
                Spacer()
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
            .menuStyle(.borderlessButton)
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

/// The draggable line between the 3D view and the chat.
struct ViewerDivider: View {
    @Binding var height: Double
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
                if start == nil { start = height }
                height = min(max(start! + v.translation.height, range.lowerBound), range.upperBound)
            }
            .onEnded { _ in start = nil })
        #if os(macOS)
        .onHover { inside in if inside { NSCursor.resizeUpDown.push() } else { NSCursor.pop() } }
        #endif
    }
}

// MARK: - SceneKit

private struct ViewerStyle {
    var background: Color
    var body: Color
    var edge: Color
    var highlight: Color
    var ambient: CGFloat

    static let light = ViewerStyle(background: Color(hex: 0xF1F1EE), body: Color(hex: 0xB9C2CC),
                                   edge: Color(hex: 0x2B323B, alpha: 0.85), highlight: Color(hex: 0x2A78D6), ambient: 420)
    static let dark = ViewerStyle(background: Color(hex: 0x141419), body: Color(hex: 0x8C97A5),
                                  edge: Color(hex: 0x0A0C0F, alpha: 0.9), highlight: Color(hex: 0x5B6CFF), ambient: 330)
}

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
        c.view = view
    }

    fileprivate func update(_ c: ViewerController) {
        c.onPick = onPick
        c.setDark(dark)
        c.show(mesh)
        c.setHighlight(highlighted)
        c.apply(command)
    }
}

#if os(macOS)
extension ModelViewer: NSViewRepresentable {
    func makeNSView(context: Context) -> PickingSCNView {
        let view = PickingSCNView(frame: .zero, options: nil)
        configure(view, context.coordinator)
        view.onClick = { [weak c = context.coordinator] in c?.pick(at: $0) }
        view.onDoubleClick = { [weak c = context.coordinator] in c?.focus(at: $0) }
        return view
    }
    func updateNSView(_ view: PickingSCNView, context: Context) { update(context.coordinator) }
}

/// SCNView's camera control owns the mouse; a press that barely moves is a click.
final class PickingSCNView: SCNView {
    var onClick: (CGPoint) -> Void = { _ in }
    var onDoubleClick: (CGPoint) -> Void = { _ in }
    private var downAt: CGPoint?

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
    private var highlightNode: SCNNode?
    private var sources: [SCNGeometrySource] = []
    private var highlighted: [Int] = []
    private var lastCommand: UUID?
    private var dark: Bool?
    private let ambient = SCNLight()
    private let bodyMaterial = SCNMaterial()
    private let edgeMaterial = SCNMaterial()
    private let highlightMaterial = SCNMaterial()

    private static let partMask = 1, overlayMask = 2

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
    }

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
        let style = dark ? ViewerStyle.dark : .light
        scene.background.contents = style.background.platform
        bodyMaterial.diffuse.contents = style.body.platform
        edgeMaterial.diffuse.contents = style.edge.platform
        highlightMaterial.diffuse.contents = style.highlight.platform
        ambient.intensity = style.ambient
    }

    func show(_ mesh: PartMesh?) {
        guard mesh?.sessionId != self.mesh?.sessionId else { return }
        partNode?.removeFromParentNode()
        partNode = nil
        highlightNode = nil
        highlighted = []
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
    }

    func setHighlight(_ faces: [Int]) {
        guard faces != highlighted else { return }
        highlighted = faces
        highlightNode?.removeFromParentNode()
        highlightNode = nil
        guard let mesh, let partNode, !faces.isEmpty else { return }
        let (data, count) = mesh.indices(ofFaces: faces)
        guard count > 0 else { return }
        let element = SCNGeometryElement(data: data, primitiveType: .triangles, primitiveCount: count, bytesPerIndex: 4)
        let geometry = SCNGeometry(sources: sources, elements: [element])
        geometry.materials = [highlightMaterial]
        let node = SCNNode(geometry: geometry)
        node.categoryBitMask = Self.overlayMask
        node.renderingOrder = 2
        partNode.addChildNode(node)
        highlightNode = node
    }

    /// Frame the whole part from a standard direction.
    func apply(_ command: ViewCommand) {
        guard command.token != lastCommand, let mesh, let view else { return }
        lastCommand = command.token
        let size = view.bounds.size
        let aspect = size.height > 0 ? Float(size.width / size.height) : 1.4
        let halfVertical = Float(cameraNode.camera?.fieldOfView ?? 30) * .pi / 360
        let halfHorizontal = atan(tan(halfVertical) * aspect)
        let distance = mesh.radius / sin(min(halfVertical, halfHorizontal)) * 1.04

        SCNTransaction.begin()
        SCNTransaction.animationDuration = command.animated ? 0.45 : 0
        SCNTransaction.animationTimingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
        cameraNode.simdPosition = mesh.center + command.preset.direction * distance
        cameraNode.simdLook(at: mesh.center, up: command.preset.up, localFront: SIMD3(0, 0, -1))
        SCNTransaction.commit()
        view.defaultCameraController.target = SCNVector3(mesh.center)
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
        cameraNode.simdLook(at: target, up: cameraNode.simdWorldUp, localFront: SIMD3(0, 0, -1))
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

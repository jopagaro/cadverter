import Foundation
#if canImport(FoundationModels)
import FoundationModels
#endif

/// Apple's on-device model (Foundation Models framework, macOS 26 / iOS 26).
///
/// Free, private, works offline. The model runs the same tool loop the hosted providers
/// use, but the loop lives here in Swift: each tool call hits the engine's `POST /tool`
/// endpoint so the numbers still come from exact B-REP geometry.
final class AppleIntelligence {
    /// Whether this OS/toolchain has the framework at all. Note this is a *build/OS* check:
    /// it says nothing about whether the hardware can run the model — Intel Macs ship macOS 26
    /// but can never run Apple Intelligence. Use `isDeviceCapable` for UI decisions.
    static var isSupportedOS: Bool {
        #if canImport(FoundationModels)
        if #available(macOS 26.0, iOS 26.0, *) { return true }
        #endif
        return false
    }

    /// Why the on-device model can or cannot answer, in the user's terms.
    ///
    /// `detail` is one line for a status row; `guidance` is the fuller explanation shown
    /// when someone tries to use it and it is not there. Every unavailable case says what
    /// to do next, and every one of them ends at the same fallback: use your own API key.
    static var status: (available: Bool, detail: String, guidance: String?) {
        #if canImport(FoundationModels)
        if #available(macOS 26.0, iOS 26.0, *) {
            switch SystemLanguageModel.default.availability {
            case .available:
                return (true, "On-device model ready — private, offline, no API key.", nil)
            case .unavailable(let reason):
                switch reason {
                case .deviceNotEligible:
                    return (false,
                            "This Mac can't run Apple Intelligence.",
                            "Apple Intelligence needs Apple silicon (M1 or later). Intel Macs "
                            + "can't run it at any macOS version, and that won't change.\n\n"
                            + "Everything else works: open a file and CADVERT analyses the exact "
                            + "geometry, renders the views and writes the document with no AI at all. "
                            + "To ask questions about a part, add your own OpenAI or Anthropic key "
                            + "below — it stays in your Keychain and is sent only to that provider.")
                case .appleIntelligenceNotEnabled:
                    return (false,
                            "Apple Intelligence is turned off.",
                            "Turn it on in System Settings → Apple Intelligence & Siri, then come "
                            + "back and choose Apple Intelligence here. It is free and runs entirely "
                            + "on this Mac.\n\nIf you would rather not, add your own OpenAI or "
                            + "Anthropic key below instead.")
                case .modelNotReady:
                    return (false,
                            "The on-device model is still downloading.",
                            "macOS is still fetching the model. This usually finishes in a few "
                            + "minutes on a good connection — it is a large download and Apple "
                            + "pauses it on battery or a metered network.\n\nTry again shortly, or "
                            + "add your own API key below to start now.")
                @unknown default:
                    return (false,
                            "Apple Intelligence is unavailable right now.",
                            "macOS reports the on-device model as unavailable without saying why. "
                            + "Check System Settings → Apple Intelligence & Siri, and that macOS is "
                            + "up to date.\n\nOr add your own API key below to continue.")
                }
            }
        }
        #endif
        return (false,
                "Needs macOS 26 or later.",
                "The on-device model arrived in macOS 26 and needs Apple silicon (M1 or later). "
                + "Update macOS if you can.\n\nAnalysis, rendered views and the geometry document "
                + "all work on this version already — only the built-in AI needs the update. To ask "
                + "questions now, add your own OpenAI or Anthropic key below.")
    }

    /// Can this Mac *ever* run the on-device model? False on hardware Apple excludes
    /// (every Intel Mac, and older devices), which is permanent and means the option
    /// should not be offered at all. Returns true when the model is merely switched off
    /// or still downloading, since the user can fix those.
    static var isDeviceCapable: Bool {
        #if canImport(FoundationModels)
        if #available(macOS 26.0, iOS 26.0, *) {
            if case .unavailable(let reason) = SystemLanguageModel.default.availability,
               reason == .deviceNotEligible {
                return false
            }
            return true
        }
        #endif
        return false
    }

    /// True only when a question asked right now would actually be answered.
    static var isReadyNow: Bool { status.available }

    private var engine: AnyObject?

    func stream(question: String, part: PartSession, client: CadvertClient) -> AsyncThrowingStream<ChatEvent, Error> {
        #if canImport(FoundationModels)
        if #available(macOS 26.0, iOS 26.0, *) {
            let e: OnDeviceEngine
            if let existing = engine as? OnDeviceEngine { e = existing } else { e = OnDeviceEngine(); engine = e }
            return e.stream(question: question, part: part, client: client)
        }
        #endif
        let detail = Self.status.detail
        return AsyncThrowingStream { $0.finish(throwing: CadvertError.stream(detail)) }
    }

    /// Forget the conversation (new part / new session).
    func reset() { engine = nil }
}

#if canImport(FoundationModels)

@available(macOS 26.0, iOS 26.0, *)
final class OnDeviceEngine {
    private var session: LanguageModelSession?
    private var sessionPartId: String?
    private var sessionBudget = 0

    /// The on-device context is ~4k tokens for everything (instructions, tool schemas,
    /// history, answer). Start with a generous summary; fall back to a tight one on overflow.
    static let summaryBudgets = [6_000, 2_400]

    func stream(question: String, part: PartSession, client: CadvertClient) -> AsyncThrowingStream<ChatEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let bridge = ToolBridge(client: client, sessionId: part.id) { continuation.yield($0) }
                    try await self.answer(question: question, part: part, bridge: bridge, continuation: continuation)
                    continuation.yield(.done)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: Self.mapError(error))
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func answer(question: String, part: PartSession, bridge: ToolBridge,
                        continuation: AsyncThrowingStream<ChatEvent, Error>.Continuation) async throws {
        var budgetIndex = 0
        var lastError: Error?
        while budgetIndex < Self.summaryBudgets.count {
            let budget = Self.summaryBudgets[budgetIndex]
            let session = makeSession(part: part, bridge: bridge, budget: budget, fresh: budgetIndex > 0)
            do {
                try await run(question: question, session: session, continuation: continuation)
                return
            } catch let e as LanguageModelSession.GenerationError {
                if case .exceededContextWindowSize = e {
                    lastError = e
                    budgetIndex += 1          // retry once with a fresh, tighter session
                    continue
                }
                throw e
            }
        }
        throw lastError ?? CadvertError.stream("The on-device model ran out of context.")
    }

    private func run(question: String, session: LanguageModelSession,
                     continuation: AsyncThrowingStream<ChatEvent, Error>.Continuation) async throws {
        var emitted = ""
        let stream = session.streamResponse(to: question)
        for try await snapshot in stream {
            let full = snapshot.content          // cumulative text so far
            guard full.count > emitted.count else { continue }
            let delta = full.hasPrefix(emitted) ? String(full.dropFirst(emitted.count)) : "\n" + full
            emitted = full
            continuation.yield(.content(delta))
        }
    }

    private func makeSession(part: PartSession, bridge: ToolBridge, budget: Int, fresh: Bool) -> LanguageModelSession {
        if !fresh, let s = session, sessionPartId == part.id, sessionBudget == budget { return s }
        let tools: [any Tool] = part.isMesh ? [] : [
            GetFeatureTool(bridge), GetFaceTool(bridge), GetEdgeTool(bridge),
            MeasureDistanceTool(bridge), GetNeighborsTool(bridge), SearchFacesTool(bridge),
        ]
        let s = LanguageModelSession(tools: tools, instructions: Self.instructions(for: part, budget: budget))
        session = s
        sessionPartId = part.id
        sessionBudget = budget
        return s
    }

    static func instructions(for part: PartSession, budget: Int) -> String {
        var summary = part.tier0 ?? part.hsd
        var note = ""
        if summary.count > budget {
            summary = String(summary.prefix(budget))
            note = "\n(The summary was truncated to fit — use the tools for anything not listed.)"
        }
        let toolRule = part.isMesh
            ? "This is a mesh file (STL/OBJ): exact B-REP geometry is unavailable — say so when asked for it."
            : "Use the tools whenever you need a number that is not in the summary: get_feature for a hole/boss/fillet/pattern by ID, get_face / get_edge for F12 / E5, measure_distance between two entities, get_neighbors for adjacent faces, search_faces to find faces by type or size."
        return """
        You are a mechanical engineer answering questions about one CAD part. All dimensions come from the exact CAD geometry; never estimate. \(toolRule)
        Reference faces as F12, edges as E5, features by ID such as hole_1. Keep answers short and precise, with units.
        <PART_SUMMARY>
        \(summary)\(note)
        </PART_SUMMARY>
        """
    }

    static func mapError(_ error: Error) -> Error {
        if let g = error as? LanguageModelSession.GenerationError {
            switch g {
            case .exceededContextWindowSize:
                return CadvertError.stream("This part is too large for the on-device model's context. Switch to OpenAI or Claude in Settings for big parts.")
            case .guardrailViolation:
                return CadvertError.stream("Apple's on-device model declined this request (safety guardrail).")
            case .rateLimited, .concurrentRequests:
                return CadvertError.stream("The on-device model is busy — try again in a moment.")
            case .assetsUnavailable:
                return CadvertError.stream("The on-device model isn't available — check Apple Intelligence in System Settings.")
            default:
                return CadvertError.stream(g.errorDescription ?? "The on-device model failed.")
            }
        }
        if let t = error as? LanguageModelSession.ToolCallError {
            return CadvertError.stream("Tool \(t.tool.name) failed: \(t.underlyingError.localizedDescription)")
        }
        return error
    }
}

// MARK: - Tools (mirror CADVERT_TOOLS in server.py; executed via POST /tool/{session})

@available(macOS 26.0, iOS 26.0, *)
final class ToolBridge: @unchecked Sendable {
    let client: CadvertClient
    let sessionId: String
    let emit: @Sendable (ChatEvent) -> Void

    init(client: CadvertClient, sessionId: String, emit: @escaping @Sendable (ChatEvent) -> Void) {
        self.client = client; self.sessionId = sessionId; self.emit = emit
    }

    func call(_ name: String, _ arguments: [String: Any]) async throws -> String {
        emit(.toolCall(name))
        return try await client.executeTool(sessionId: sessionId, name: name, arguments: arguments)
    }
}

@available(macOS 26.0, iOS 26.0, *)
struct GetFeatureTool: Tool {
    let name = "get_feature"
    let description = "Get full geometric detail for a feature (hole, boss, fillet, countersink, pattern…): all faces with exact surface parameters, boundary edges and measurements."
    @Generable struct Arguments {
        @Guide(description: "Feature ID from the part summary, e.g. hole_1, fillet_3, pattern_1")
        var feature_id: String
    }
    let bridge: ToolBridge
    init(_ bridge: ToolBridge) { self.bridge = bridge }
    func call(arguments: Arguments) async throws -> String {
        try await bridge.call(name, ["feature_id": arguments.feature_id])
    }
}

@available(macOS 26.0, iOS 26.0, *)
struct GetFaceTool: Tool {
    let name = "get_face"
    let description = "Get exact geometry for one B-REP face: surface type, parameters (normal, radius, axis…), area, and its boundary edges with dihedral angles."
    @Generable struct Arguments {
        @Guide(description: "Face ID, e.g. F12 or 12")
        var face_id: String
    }
    let bridge: ToolBridge
    init(_ bridge: ToolBridge) { self.bridge = bridge }
    func call(arguments: Arguments) async throws -> String {
        try await bridge.call(name, ["face_id": arguments.face_id])
    }
}

@available(macOS 26.0, iOS 26.0, *)
struct GetEdgeTool: Tool {
    let name = "get_edge"
    let description = "Get exact geometry for one edge: curve type, length, the two faces it connects and the dihedral angle between them."
    @Generable struct Arguments {
        @Guide(description: "Edge ID, e.g. E5 or 5")
        var edge_id: String
    }
    let bridge: ToolBridge
    init(_ bridge: ToolBridge) { self.bridge = bridge }
    func call(arguments: Arguments) async throws -> String {
        try await bridge.call(name, ["edge_id": arguments.edge_id])
    }
}

@available(macOS 26.0, iOS 26.0, *)
struct MeasureDistanceTool: Tool {
    let name = "measure_distance"
    let description = "Compute the exact minimum distance between two entities (faces like F12, features like hole_1, or points x,y,z) and the closest points on each."
    @Generable struct Arguments {
        @Guide(description: "First entity: face ID like F12, feature like hole_1, or a point x,y,z")
        var entity_a: String
        @Guide(description: "Second entity, same format as entity_a")
        var entity_b: String
    }
    let bridge: ToolBridge
    init(_ bridge: ToolBridge) { self.bridge = bridge }
    func call(arguments: Arguments) async throws -> String {
        try await bridge.call(name, ["entity_a": arguments.entity_a, "entity_b": arguments.entity_b])
    }
}

@available(macOS 26.0, iOS 26.0, *)
struct GetNeighborsTool: Tool {
    let name = "get_neighbors"
    let description = "Get all faces adjacent to a face within N edge hops, with their geometry and the connecting edges."
    @Generable struct Arguments {
        @Guide(description: "Starting face ID, e.g. F12")
        var face_id: String
        @Guide(description: "Number of edge hops to traverse; 1 means immediate neighbours")
        var depth: Int
    }
    let bridge: ToolBridge
    init(_ bridge: ToolBridge) { self.bridge = bridge }
    func call(arguments: Arguments) async throws -> String {
        try await bridge.call(name, ["face_id": arguments.face_id, "depth": max(1, arguments.depth)])
    }
}

@available(macOS 26.0, iOS 26.0, *)
struct SearchFacesTool: Tool {
    let name = "search_faces"
    let description = "Find faces matching criteria: surface type and/or radius or area ranges."
    @Generable struct Arguments {
        @Guide(description: "Surface type filter: plane, cylinder, cone, sphere, torus or nurbs. Leave empty for any.")
        var surface_type: String
        @Guide(description: "Minimum radius in part units, or 0 for no minimum")
        var radius_min: Double
        @Guide(description: "Maximum radius in part units, or 0 for no maximum")
        var radius_max: Double
    }
    let bridge: ToolBridge
    init(_ bridge: ToolBridge) { self.bridge = bridge }
    func call(arguments: Arguments) async throws -> String {
        var args: [String: Any] = [:]
        let t = arguments.surface_type.trimmingCharacters(in: .whitespaces).lowercased()
        if ["plane", "cylinder", "cone", "sphere", "torus", "nurbs"].contains(t) { args["surface_type"] = t }
        if arguments.radius_min > 0 { args["radius_min"] = arguments.radius_min }
        if arguments.radius_max > 0 { args["radius_max"] = arguments.radius_max }
        return try await bridge.call(name, args)
    }
}

#endif

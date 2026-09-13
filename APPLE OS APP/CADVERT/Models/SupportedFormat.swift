import Foundation
import UniformTypeIdentifiers

/// The formats the web UI accepts (its `<input accept>` list), with the same full/mesh split.
enum SupportedFormat: String, CaseIterable {
    case step, stp, iges, igs, brep, stl, obj

    /// Full B-REP analysis (features, spatial, GD&T) vs. mesh-only.
    var isFullAnalysis: Bool {
        switch self {
        case .step, .stp, .iges, .igs, .brep: return true
        case .stl, .obj: return false
        }
    }

    static var allExtensions: [String] { allCases.map(\.rawValue) }

    static func from(filename: String) -> SupportedFormat? {
        let ext = (filename as NSString).pathExtension.lowercased()
        return SupportedFormat(rawValue: ext)
    }

    // Uniform types for the file importer / drop targets.
    static let stepType = UTType(importedAs: "com.cadvert.step")
    static let igesType = UTType(importedAs: "com.cadvert.iges")
    static let brepType = UTType(importedAs: "com.cadvert.brep")
    static let stlType  = UTType("public.standard-tesselated-geometry-format") ?? .data
    static let objType  = UTType("public.geometry-definition-format") ?? .data

    static var contentTypes: [UTType] { [stepType, igesType, brepType, stlType, objType] }
}

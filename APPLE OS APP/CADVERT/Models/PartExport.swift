import Foundation
import UniformTypeIdentifiers

/// What the part can be saved as — the kinds `GET /export/{session}/{kind}` writes.
enum PartExport: String, CaseIterable, Identifiable {
    case stl, threeMF = "3mf", obj, glb, step, bom = "bom.csv", features = "features.csv"

    var id: String { rawValue }

    /// The part as a file, then its tables.
    static let files: [PartExport] = [.stl, .threeMF, .obj, .glb, .step]
    static let tables: [PartExport] = [.bom, .features]

    var menuTitle: String {
        switch self {
        case .stl: return "STL for 3D Printing…"
        case .threeMF: return "3MF for 3D Printing…"
        case .obj: return "OBJ…"
        case .glb: return "GLB for the Web…"
        case .step: return "STEP…"
        case .bom: return "Bill of Materials (CSV)…"
        case .features: return "Feature Table (CSV)…"
        }
    }

    /// Names the file in "STL saved" and "Exporting STL…".
    var shortName: String {
        switch self {
        case .bom: return "Bill of materials"
        case .features: return "Feature table"
        case .threeMF: return "3MF"
        default: return rawValue.uppercased()
        }
    }

    func filename(for stem: String) -> String {
        switch self {
        case .bom: return "\(stem) BOM.csv"
        case .features: return "\(stem) features.csv"
        default: return "\(stem).\(rawValue)"
        }
    }

    var contentType: UTType {
        switch self {
        case .stl: return SupportedFormat.stlType
        case .obj: return SupportedFormat.objType
        case .step: return SupportedFormat.stepType
        case .bom, .features: return .commaSeparatedText
        case .threeMF, .glb: return UTType(filenameExtension: rawValue, conformingTo: .data) ?? .data
        }
    }

    /// Whether this part can be written this way: STEP only for a solid that didn't come in
    /// as STEP, the bill of materials only for an assembly, the feature table only when
    /// features were found.
    func isAvailable(for session: PartSession) -> Bool {
        switch self {
        case .stl, .threeMF, .obj, .glb: return true
        case .step: return !session.isMesh && session.format.uppercased() != "STEP"
        case .bom: return (session.summary.components ?? 0) > 1
        case .features: return !session.isMesh && !(session.summary.features ?? [:]).isEmpty
        }
    }
}

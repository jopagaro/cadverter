import Foundation
import simd

/// The part as the engine tessellated it for the 3D view (`GET /mesh/{session}`), every
/// triangle tagged with the B-REP face it came from. Layout: `src/cadvert/viewmesh.py`.
struct PartMesh {
    var sessionId: String
    var vertexCount: Int
    var triangleCount: Int
    var edgePointCount: Int
    var edgeSegmentCount: Int
    var positions: Data        // float32 × 3 per vertex
    var normals: Data          // float32 × 3 per vertex
    var indices: Data          // uint32 × 3 per triangle
    var edgePoints: Data       // float32 × 3 per point
    var edgeSegments: Data     // uint32 × 2 per segment
    /// Per face (ID = index + 1): first triangle and triangle count. Faces are contiguous
    /// and in ID order, which is what makes `face(ofTriangle:)` a binary search.
    var faceRanges: [(first: Int, count: Int)]
    var boundsMin: SIMD3<Float>
    var boundsMax: SIMD3<Float>
    var units: String

    var center: SIMD3<Float> { (boundsMin + boundsMax) / 2 }
    var radius: Float { max(simd_length(boundsMax - boundsMin) / 2, 1e-3) }

    enum ParseError: Error { case badMagic, truncated, badHeader }

    init(data: Data, sessionId: String) throws {
        self.sessionId = sessionId
        guard data.count >= 8, data.prefix(4) == Data("CVM1".utf8) else { throw ParseError.badMagic }
        let headerLength = Int(data.uint32(at: 4))
        guard data.count >= 8 + headerLength else { throw ParseError.truncated }

        struct Header: Decodable {
            var units: String?
            var vertices: Int
            var triangles: Int
            var faces: Int
            var edge_points: Int
            var edge_segments: Int
            var bbox: [Double]
        }
        guard let h = try? JSONDecoder().decode(Header.self, from: data.subdata(in: 8 ..< 8 + headerLength)),
              h.bbox.count == 6 else { throw ParseError.badHeader }

        var offset = 8 + headerLength
        func take(_ bytes: Int) throws -> Data {
            guard bytes >= 0, offset + bytes <= data.count else { throw ParseError.truncated }
            defer { offset += bytes }
            return data.subdata(in: offset ..< offset + bytes)
        }
        positions = try take(h.vertices * 12)
        normals = try take(h.vertices * 12)
        indices = try take(h.triangles * 12)
        let ranges = try take(h.faces * 8)
        edgePoints = try take(h.edge_points * 12)
        edgeSegments = try take(h.edge_segments * 8)

        vertexCount = h.vertices
        triangleCount = h.triangles
        edgePointCount = h.edge_points
        edgeSegmentCount = h.edge_segments
        faceRanges = (0 ..< h.faces).map { i in
            (Int(ranges.uint32(at: i * 8)), Int(ranges.uint32(at: i * 8 + 4)))
        }
        boundsMin = SIMD3(Float(h.bbox[0]), Float(h.bbox[1]), Float(h.bbox[2]))
        boundsMax = SIMD3(Float(h.bbox[3]), Float(h.bbox[4]), Float(h.bbox[5]))
        units = h.units ?? "mm"
    }

    /// The face (1-based ID) a triangle belongs to.
    func face(ofTriangle t: Int) -> Int? {
        guard t >= 0, t < triangleCount else { return nil }
        var lo = 0, hi = faceRanges.count - 1
        while lo <= hi {
            let mid = (lo + hi) / 2
            let r = faceRanges[mid]
            if t < r.first { hi = mid - 1 }
            else if t >= r.first + r.count { lo = mid + 1 }
            else { return mid + 1 }
        }
        return nil
    }

    /// The triangle indices of some faces, packed for a highlight overlay.
    func indices(ofFaces ids: [Int]) -> (data: Data, triangles: Int) {
        var out = Data()
        var total = 0
        for id in Set(ids).sorted() where id >= 1 && id <= faceRanges.count {
            let r = faceRanges[id - 1]
            guard r.count > 0 else { continue }
            out.append(indices.subdata(in: r.first * 12 ..< (r.first + r.count) * 12))
            total += r.count
        }
        return (out, total)
    }
}

private extension Data {
    func uint32(at offset: Int) -> UInt32 {
        var v: UInt32 = 0
        Swift.withUnsafeMutableBytes(of: &v) { dst in
            dst.copyBytes(from: self[(startIndex + offset) ..< (startIndex + offset + 4)])
        }
        return UInt32(littleEndian: v)
    }
}

// MARK: - Scale guide

extension PartMesh {
    var size: SIMD3<Float> { boundsMax - boundsMin }

    /// Grid spacing for the scale guide: the round number (1, 2 or 5 × 10ⁿ) nearest to a
    /// tenth of the part's footprint, so there are about ten squares across it.
    var gridStep: Float {
        let raw = max(size.x, size.y, 1e-3) / 10
        let magnitude = pow(10, floor(log10(raw)))
        return [1, 2, 5, 10].map { $0 * magnitude }.min { abs(log($0 / raw)) < abs(log($1 / raw)) }!
    }

    /// A coordinate on an axis: like `format`, with a proper minus sign below zero.
    static func format(_ v: Float, signed: Bool) -> String {
        guard signed, v < 0 else { return format(abs(v) < 1e-6 ? 0 : v) }
        let s = format(-v)
        return s == "0" ? s : "−" + s
    }

    /// A length as people write it: "140", "17.5", "6.35" — no trailing zeros.
    static func format(_ v: Float) -> String {
        let decimals = v >= 100 ? 0 : v >= 10 ? 1 : 2
        var s = String(format: "%.\(decimals)f", v)
        if s.contains(".") {
            while s.hasSuffix("0") { s.removeLast() }
            if s.hasSuffix(".") { s.removeLast() }
        }
        return s
    }
}

/// The standard views, named and aimed as the engine's rendered views are (Z up): the
/// direction is where the camera sits relative to the part's centre.
enum ViewPreset: String, CaseIterable, Identifiable {
    case iso, front, top, right, back, bottom, left

    var id: String { rawValue }
    var label: String { rawValue.capitalized }

    var direction: SIMD3<Float> {
        switch self {
        case .iso:    return simd_normalize(SIMD3(1, -1, 1))
        case .front:  return SIMD3(0, -1, 0)
        case .back:   return SIMD3(0, 1, 0)
        case .left:   return SIMD3(-1, 0, 0)
        case .right:  return SIMD3(1, 0, 0)
        case .top:    return SIMD3(0, 0, 1)
        case .bottom: return SIMD3(0, 0, -1)
        }
    }

    var up: SIMD3<Float> {
        switch self {
        case .top, .bottom: return SIMD3(0, 1, 0)
        default:            return SIMD3(0, 0, 1)
        }
    }
}

/// A one-shot request to move the 3D view's camera; the token makes repeats count.
struct ViewCommand: Equatable {
    var preset: ViewPreset
    var animated = true
    var token = UUID()
}

enum MeshState: Equatable {
    case none, loading, ready
    case failed(String)
}

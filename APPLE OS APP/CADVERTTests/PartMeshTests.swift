import XCTest
import simd
@testable import CADVERT

final class PartMeshTests: XCTestCase {
    /// Two faces of one triangle each, and one edge — packed as the engine packs it.
    private func packed(magic: String = "CVM1", truncate: Int = 0) -> Data {
        let header = #"{"version": 1, "units": "mm", "vertices": 4, "triangles": 2, "faces": 2, "edge_points": 2, "edge_segments": 1, "bbox": [0, 0, 0, 140, 90, 18]}"#
        var head = Data(header.utf8)
        while head.count % 4 != 0 { head.append(0x20) }
        var d = Data(magic.utf8)
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        func f32(_ v: Float) { withUnsafeBytes(of: v.bitPattern.littleEndian) { d.append(contentsOf: $0) } }
        u32(UInt32(head.count)); d.append(head)
        for v in [0, 0, 0, 140, 0, 0, 140, 90, 0, 0, 90, 18] as [Float] { f32(v) }   // positions
        for _ in 0 ..< 4 { f32(0); f32(0); f32(1) }                                       // normals
        for i in [0, 1, 2, 0, 2, 3] as [UInt32] { u32(i) }                               // triangles
        for r in [0, 1, 1, 1] as [UInt32] { u32(r) }                                     // face ranges
        for v in [0, 0, 0, 140, 0, 0] as [Float] { f32(v) }                              // edge points
        u32(0); u32(1)                                                                   // edge segment
        return d.dropLast(truncate)
    }

    func testReadsTheEngineLayout() throws {
        let m = try PartMesh(data: packed(), sessionId: "s")
        XCTAssertEqual(m.vertexCount, 4)
        XCTAssertEqual(m.triangleCount, 2)
        XCTAssertEqual(m.edgeSegmentCount, 1)
        XCTAssertEqual(m.faceRanges.map(\.first), [0, 1])
        XCTAssertEqual(m.size, SIMD3(140, 90, 18))
        XCTAssertEqual(m.center, SIMD3(70, 45, 9))
    }

    func testTrianglesMapBackToFaces() throws {
        let m = try PartMesh(data: packed(), sessionId: "s")
        XCTAssertEqual(m.face(ofTriangle: 0), 1)
        XCTAssertEqual(m.face(ofTriangle: 1), 2)
        XCTAssertNil(m.face(ofTriangle: 2))
        XCTAssertEqual(m.indices(ofFaces: [2]).triangles, 1)
        XCTAssertEqual(m.indices(ofFaces: [1, 2, 2, 99]).triangles, 2)
    }

    func testRegionOfSomeFaces() throws {
        let m = try PartMesh(data: packed(), sessionId: "s")
        // Face 1 is the triangle (0,0,0) (140,0,0) (140,90,0): flat on the floor, facing up.
        let one = try XCTUnwrap(m.region(ofFaces: [1]))
        XCTAssertEqual(one.center, SIMD3(70, 45, 0))
        XCTAssertEqual(one.radius, simd_length(SIMD3<Float>(140, 90, 0)) / 2, accuracy: 1e-3)
        XCTAssertEqual(one.facing.z, 1, accuracy: 1e-5)
        XCTAssertEqual(try XCTUnwrap(m.region(ofFaces: [1, 2])).center, m.center)
        XCTAssertNil(m.region(ofFaces: [99]))
        XCTAssertNil(m.region(ofFaces: []))
    }

    func testRejectsBadData() {
        XCTAssertThrowsError(try PartMesh(data: packed(magic: "NOPE"), sessionId: "s"))
        XCTAssertThrowsError(try PartMesh(data: packed(truncate: 4), sessionId: "s"))
        XCTAssertThrowsError(try PartMesh(data: Data(), sessionId: "s"))
    }

    func testGridStepIsARoundNumber() throws {
        let plate = try PartMesh(data: packed(), sessionId: "s")
        XCTAssertEqual(plate.gridStep, 10)                  // 140 mm across → 10 mm squares
        var big = plate
        big.boundsMax = SIMD3(448, 205, 788)               // the T2M assembly's footprint
        XCTAssertEqual(big.gridStep, 50)
        big.boundsMax = SIMD3(3.2, 1, 1)
        XCTAssertEqual(big.gridStep, 0.5, accuracy: 1e-6)
    }

    func testLengthsReadNaturally() {
        XCTAssertEqual(PartMesh.format(140.0000002), "140")
        XCTAssertEqual(PartMesh.format(17.5), "17.5")
        XCTAssertEqual(PartMesh.format(6.35), "6.35")
        XCTAssertEqual(PartMesh.format(90), "90")
        XCTAssertEqual(PartMesh.format(0.5), "0.5")
        XCTAssertEqual(PartMesh.format(-150, signed: true), "−150")   // a real minus sign on the axes
        XCTAssertEqual(PartMesh.format(-0.0001, signed: true), "0")
        XCTAssertEqual(PartMesh.format(50, signed: true), "50")
    }
}

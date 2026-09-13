import XCTest
@testable import CADVERT

final class ModelsDecodingTests: XCTestCase {
    let convertJSON = """
    {
      "session_id": "abc-123",
      "hsd": "PART: bracket\\nUNITS: mm",
      "images": [
        {"name": "bracket_front", "url": "/sessions/abc-123/views/bracket_front.png"},
        {"name": "bracket_iso", "url": "/sessions/abc-123/views/bracket_iso.png"}
      ],
      "format": "STEP",
      "is_mesh": false,
      "filename": "bracket.step",
      "units": "mm",
      "summary": {
        "format": "STEP", "schema": "AP214", "units": "mm", "is_mesh": false,
        "faces": 8, "edges": 18, "bodies": 1, "gdt_count": 2,
        "features": {"THROUGH_HOLE": 2, "PATTERN": 1}
      }
    }
    """

    func testConvertResponseDecodes() throws {
        let r = try JSONDecoder().decode(ConvertResponse.self, from: Data(convertJSON.utf8))
        XCTAssertEqual(r.sessionId, "abc-123")
        XCTAssertEqual(r.images.count, 2)
        XCTAssertEqual(r.images[1].label, "iso")
        XCTAssertEqual(r.summary.faces, 8)
        XCTAssertEqual(r.summary.features?["THROUGH_HOLE"], 2)
        XCTAssertEqual(r.summary.sortedFeatures.first?.name, "THROUGH_HOLE")

        let session = PartSession(r)
        XCTAssertEqual(session.stem, "bracket")
        XCTAssertEqual(session.systemSummaryLine,
                       "bracket.step processed · 8 faces · 18 edges · mm · 2 through hole, 1 pattern")
    }

    func testImageURLResolvesAgainstBase() throws {
        let r = try JSONDecoder().decode(ConvertResponse.self, from: Data(convertJSON.utf8))
        let base = URL(string: "http://127.0.0.1:8765")!
        XCTAssertEqual(r.images[0].resolvedURL(base: base)?.absoluteString,
                       "http://127.0.0.1:8765/sessions/abc-123/views/bracket_front.png")
    }

    func testMeshSummaryLine() throws {
        var r = try JSONDecoder().decode(ConvertResponse.self, from: Data(convertJSON.utf8))
        r.isMesh = true; r.format = "STL"; r.filename = "part.stl"
        XCTAssertEqual(PartSession(r).systemSummaryLine, "part.stl · STL (mesh) · Ask about shape and size.")
    }

    func testServerConfigDecodes() throws {
        let cfg = try JSONDecoder().decode(ServerConfig.self,
            from: Data(#"{"disable_auth": true, "stripe_enabled": false, "stripe_byok_enabled": false}"#.utf8))
        XCTAssertTrue(cfg.disableAuth)
        XCTAssertFalse(cfg.stripeEnabled)
    }

    func testErrorMapping() {
        let plain = Data(#"{"detail": "File too large"}"#.utf8)
        XCTAssertEqual(CadvertError.from(status: 413, data: plain), .http(status: 413, message: "File too large"))

        let byok = Data(#"{"detail": {"error": "byok_required", "message": "Upgrade to continue.", "messages_used": 3}}"#.utf8)
        XCTAssertEqual(CadvertError.from(status: 429, data: byok), .byokRequired("Upgrade to continue."))

        let keyOnly = Data(#"{"detail": {"error": "byok_key_required", "message": "Enter your key"}}"#.utf8)
        XCTAssertEqual(CadvertError.from(status: 429, data: keyOnly), .byokKeyRequired("Enter your key"))

        let fileLimit = Data(#"{"detail": {"error": "file_limit", "message": "Daily limit"}}"#.utf8)
        XCTAssertEqual(CadvertError.from(status: 429, data: fileLimit), .fileLimit("Daily limit"))

        XCTAssertEqual(CadvertError.from(status: 401, data: Data()), .unauthorized("unauthorized"))
        XCTAssertEqual(CadvertError.from(status: 503, data: Data(#"{"detail": "Server API key not configured."}"#.utf8)),
                       .serverKeyMissing("Server API key not configured."))
    }

    func testSupportedFormats() {
        XCTAssertEqual(SupportedFormat.from(filename: "Part.STP"), .stp)
        XCTAssertNil(SupportedFormat.from(filename: "drawing.dxf"))
        XCTAssertTrue(SupportedFormat.step.isFullAnalysis)
        XCTAssertFalse(SupportedFormat.stl.isFullAnalysis)
    }

    func testProcessingStagesMatchWebUI() {
        XCTAssertEqual(ProcessingStages.label(elapsed: 0), "Uploading file…")
        XCTAssertEqual(ProcessingStages.label(elapsed: 2), "Loading geometry…")
        XCTAssertEqual(ProcessingStages.label(elapsed: 9), "Building topology graph…")
        XCTAssertEqual(ProcessingStages.label(elapsed: 31), "Detecting features…")
        XCTAssertEqual(ProcessingStages.label(elapsed: 40), "Computing spatial relationships…")
        XCTAssertEqual(ProcessingStages.label(elapsed: 100), "Rendering views…")
        XCTAssertEqual(ProcessingStages.label(elapsed: 200), "Almost there…")
        XCTAssertEqual(ProcessingStages.percent(elapsed: 0), 5)
        XCTAssertEqual(ProcessingStages.percent(elapsed: 10), 14)
        XCTAssertEqual(ProcessingStages.percent(elapsed: 1000), 95)
    }

    func testToolLabels() {
        XCTAssertEqual(ToolLabels.label(for: "measure_distance"), "Computing exact distance")
        XCTAssertEqual(ToolLabels.label(for: "mystery"), "Calling mystery")
    }
}

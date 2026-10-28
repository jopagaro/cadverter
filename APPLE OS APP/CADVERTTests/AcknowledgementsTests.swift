import XCTest
@testable import CADVERT

/// The Acknowledgements screen is a legal notice, not decoration: it has to ship, decode, and
/// carry the license text for every component. These catch a missing resource or a manifest
/// edit that leaves a component without its license.
final class AcknowledgementsTests: XCTestCase {

    private func load() throws -> Acknowledgements {
        try XCTUnwrap(Acknowledgements.load(), "Acknowledgements.json is missing from the app bundle")
    }

    func testOpenCASCADENoticeComesFirst() throws {
        // The Open CASCADE exception asks for a prominent notice that the software uses OCCT.
        let first = try XCTUnwrap(try load().components.first)
        XCTAssertEqual(first.name, "Open CASCADE Technology")
        XCTAssertTrue(first.notice.contains("makes use of facilities provided by the Open CASCADE Technology software"))
        XCTAssertTrue(first.texts.contains { $0.body.contains("GNU LESSER GENERAL PUBLIC LICENSE") })
        XCTAssertTrue(first.texts.contains { $0.body.contains("Open CASCADE exception") })
    }

    func testEveryComponentCarriesItsLicense() throws {
        for c in try load().components {
            if c.license == "Not declared" {
                XCTAssertFalse(c.notice.isEmpty, "\(c.name): an undeclared license needs an explanation")
            } else {
                XCTAssertFalse(c.texts.isEmpty, "\(c.name) has no license text")
                XCTAssertFalse(c.texts.contains { $0.body.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty },
                               "\(c.name) has an empty license text")
            }
        }
    }

    func testListsWhatShipsAndNothingThatDoesnt() throws {
        let names = Set(try load().components.map { $0.name.lowercased() })
        for expected in ["python", "numpy", "vtk", "cadquery-ocp", "freetype", "libraw", "freeimage"] {
            XCTAssertTrue(names.contains(expected), "\(expected) ships but is not acknowledged")
        }
        // pip is stripped from the shipped engine, and cadvert is our own code.
        XCTAssertFalse(names.contains("pip"))
        XCTAssertFalse(names.contains("cadvert"))
    }

    func testEveryComponentBelongsToAListedGroup() throws {
        let acks = try load()
        let total = acks.sections().reduce(0) { $0 + $1.items.count }
        XCTAssertEqual(total, acks.components.count, "a component's group is missing from `groups`, so it would never show")
    }

    func testSearchMatchesNameAndLicense() throws {
        let acks = try load()
        let byName = acks.sections(matching: "freetype").flatMap(\.items).map(\.name)
        XCTAssertEqual(byName, ["FreeType"])
        let byLicense = acks.sections(matching: "LGPL").flatMap(\.items).map(\.name)
        XCTAssertTrue(byLicense.contains("Open CASCADE Technology"))
        XCTAssertTrue(byLicense.contains("LibRaw"))
        XCTAssertTrue(acks.sections(matching: "no-such-component-xyz").isEmpty)
    }
}

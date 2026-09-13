import XCTest
@testable import CADVERT

final class MarkdownBlocksTests: XCTestCase {
    func testParagraphsAndCode() {
        let text = "The wall is **11.000 mm**.\n\n```\nF1 PLANE x=0\n```\nDone."
        let blocks = MarkdownBlocks.parse(text)
        XCTAssertEqual(blocks, [
            .paragraph("The wall is **11.000 mm**."),
            .code("F1 PLANE x=0"),
            .paragraph("Done."),
        ])
    }

    func testLanguageHintStripped() {
        XCTAssertEqual(MarkdownBlocks.parse("```python\nprint(1)\n```"), [.code("print(1)")])
    }

    func testLists() {
        let text = "Holes:\n- hole_1: d=8.000mm\n- hole_2: d=8.000mm\n\nTwo total."
        XCTAssertEqual(MarkdownBlocks.parse(text), [
            .paragraph("Holes:"),
            .list(["hole_1: d=8.000mm", "hole_2: d=8.000mm"]),
            .paragraph("Two total."),
        ])
    }

    func testNumberedList() {
        XCTAssertEqual(MarkdownBlocks.parse("1. first\n2. second"), [.list(["first", "second"])])
    }

    func testEmpty() {
        XCTAssertEqual(MarkdownBlocks.parse(""), [])
        XCTAssertEqual(MarkdownBlocks.parse("\n\n"), [])
    }
}

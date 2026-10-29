import XCTest
@testable import CADVERT

final class SSEParserTests: XCTestCase {
    func testContentLine() {
        XCTAssertEqual(SSEParser.parse(line: #"data: {"content": "Hello"}"#), .content("Hello"))
    }

    func testToolCallLine() {
        XCTAssertEqual(SSEParser.parse(line: #"data: {"tool_call": "get_feature"}"#), .toolCall("get_feature"))
    }

    func testHighlightLine() {
        XCTAssertEqual(SSEParser.parse(line: #"data: {"highlight": {"faces": [12, 30], "label": "Thinnest wall · 1.6 mm"}}"#),
                       .highlight(ViewerHighlight(faces: [12, 30], label: "Thinnest wall · 1.6 mm")))
        XCTAssertEqual(SSEParser.parse(line: #"data: {"highlight": {"faces": [4]}}"#),
                       .highlight(ViewerHighlight(faces: [4], label: "")))
        XCTAssertNil(SSEParser.parse(line: #"data: {"highlight": {"faces": []}}"#))
    }

    func testErrorLine() {
        XCTAssertEqual(SSEParser.parse(line: #"data: {"error": "boom"}"#), .error("boom"))
    }

    func testDone() {
        XCTAssertEqual(SSEParser.parse(line: "data: [DONE]"), .done)
    }

    func testIgnoresNonDataAndPartialJSON() {
        XCTAssertNil(SSEParser.parse(line: ""))
        XCTAssertNil(SSEParser.parse(line: "event: ping"))
        XCTAssertNil(SSEParser.parse(line: #"data: {"content": "unterminated"#))
    }

    func testTrailingNewlineTolerated() {
        XCTAssertEqual(SSEParser.parse(line: "data: {\"content\": \"x\"}\r"), .content("x"))
    }
}

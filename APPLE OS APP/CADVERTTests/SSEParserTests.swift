import XCTest
@testable import CADVERT

final class SSEParserTests: XCTestCase {
    func testContentLine() {
        XCTAssertEqual(SSEParser.parse(line: #"data: {"content": "Hello"}"#), .content("Hello"))
    }

    func testToolCallLine() {
        XCTAssertEqual(SSEParser.parse(line: #"data: {"tool_call": "get_feature"}"#), .toolCall("get_feature"))
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

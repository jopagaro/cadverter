import XCTest
@testable import CADVERT

final class MarkdownBlocksTests: XCTestCase {
    private typealias Item = MarkdownBlock.ListItem

    func testParagraphsAndCode() {
        let text = "The wall is **11.000 mm**.\n\n```\nF1 PLANE x=0\n```\nDone."
        XCTAssertEqual(MarkdownBlocks.parse(text), [
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
            .list([Item(marker: nil, text: "hole_1: d=8.000mm"), Item(marker: nil, text: "hole_2: d=8.000mm")]),
            .paragraph("Two total."),
        ])
    }

    func testNumberedListKeepsNumbers() {
        XCTAssertEqual(MarkdownBlocks.parse("1. first\n2) second"),
                       [.list([Item(marker: "1.", text: "first"), Item(marker: "2.", text: "second")])])
    }

    func testNestedListAndContinuation() {
        let text = "- Bores\n  - ∅20 through\n    seated bearings\n- Taps"
        XCTAssertEqual(MarkdownBlocks.parse(text), [.list([
            Item(marker: nil, text: "Bores"),
            Item(marker: nil, text: "∅20 through seated bearings", depth: 1),
            Item(marker: nil, text: "Taps"),
        ])])
    }

    func testHeadingsRuleAndQuote() {
        let text = "## Hole census ##\nText\n\n---\n> Measured, not estimated."
        XCTAssertEqual(MarkdownBlocks.parse(text), [
            .heading(level: 2, text: "Hole census"),
            .paragraph("Text"),
            .rule,
            .quote("Measured, not estimated."),
        ])
        // "#hashtag" without a space is not a heading.
        XCTAssertEqual(MarkdownBlocks.parse("#5 hole"), [.paragraph("#5 hole")])
    }

    func testTable() {
        let text = """
        Most common sizes:
        | ∅ (mm) | Count | Use |
        |---|:---:|---|
        | 4.134 | 42 | M5 tap |
        | 9.000 | 36 | M8 clearance \\| blind |
        After.
        """
        XCTAssertEqual(MarkdownBlocks.parse(text), [
            .paragraph("Most common sizes:"),
            .table(.init(header: ["∅ (mm)", "Count", "Use"],
                         rows: [["4.134", "42", "M5 tap"], ["9.000", "36", "M8 clearance | blind"]],
                         // first column stays leading; "Count" was centred explicitly
                         alignments: [.leading, .center, .leading])),
            .paragraph("After."),
        ])
    }

    func testTableNumbersRightAlignAndRowsAreSquared() {
        let text = "| Feature | ∅ | Depth |\n|---|---|---|\n| hole_1 | ∅6.600 mm | 14 |\n| hole_2 | ∅4.2 mm |"
        guard case .table(let t) = MarkdownBlocks.parse(text).first else { return XCTFail("no table") }
        XCTAssertEqual(t.alignments, [.leading, .trailing, .trailing])
        XCTAssertEqual(t.rows[1], ["hole_2", "∅4.2 mm", ""])
    }

    func testQuantityLedTextStaysLeftAligned() {
        let text = """
        | Subsystem | Parts found | Force |
        |---|---|---|
        | Spindle drive | 2× 12 V / 3500 rpm motors, 2× S5M-300 belt | 480 N |
        | Feed axis | 8× LBHSW20 blocks on ∅20 shafts | 1,200 N |
        """
        guard case .table(let t) = MarkdownBlocks.parse(text).first else { return XCTFail("no table") }
        XCTAssertEqual(t.alignments, [.leading, .leading, .trailing])
    }

    func testNumberCells() {
        for yes in ["42", "∅6.600 mm", "12×", "77%", "1,200 N", "-3.5", "90°", "**14**"] {
            XCTAssertTrue(MarkdownBlocks.isNumberCell(yes), yes)
        }
        for no in ["2× 12 V motors", "M5 tap", "hole_9", "4.134 tap drill", ""] {
            XCTAssertFalse(MarkdownBlocks.isNumberCell(no), no)
        }
    }

    func testTableWithoutSeparatorIsJustText() {
        // Mid-stream: the separator row hasn't arrived yet.
        XCTAssertEqual(MarkdownBlocks.parse("| a | b |"), [.paragraph("| a | b |")])
    }

    func testChartBlocks() {
        let json = #"{"type": "bar", "labels": ["A", "B"], "values": [3, 5]}"#
        guard case .chart(let spec) = MarkdownBlocks.parse("```chart\n\(json)\n```").first else {
            return XCTFail("no chart")
        }
        XCTAssertEqual(spec.kind, .bar)
        XCTAssertEqual(spec.labels, ["A", "B"])
        XCTAssertEqual(spec.series.first?.values, [3, 5])

        // Still streaming: placeholder, never half a JSON object.
        XCTAssertEqual(MarkdownBlocks.parse("Here:\n```chart\n{\"type\": \"ba"), [.paragraph("Here:"), .chartPending])
        // Unusable JSON falls back to showing the code.
        XCTAssertEqual(MarkdownBlocks.parse("```chart\nnot json\n```"), [.code("not json")])
    }

    func testEmpty() {
        XCTAssertEqual(MarkdownBlocks.parse(""), [])
        XCTAssertEqual(MarkdownBlocks.parse("\n\n"), [])
    }
}

final class ChartSpecTests: XCTestCase {
    func testAliasesAndShapes() {
        let a = ChartSpec.decode(#"{"type": "pie", "data": [{"label": "Steel", "value": 3}, {"label": "Brass", "value": "1"}]}"#)
        XCTAssertEqual(a?.kind, .donut)
        XCTAssertEqual(a?.labels, ["Steel", "Brass"])
        XCTAssertEqual(a?.series.first?.values, [3, 1])

        let b = ChartSpec.decode(#"{"type": "vbar", "x": "Size", "categories": [4.1, 6], "series": [{"name": "Holes", "values": [2, 7]}]}"#)
        XCTAssertEqual(b?.kind, .column)
        XCTAssertEqual(b?.xLabel, "Size")
        XCTAssertEqual(b?.labels, ["4.1", "6"])

        XCTAssertNil(ChartSpec.decode(#"{"type": "radar", "labels": ["a"], "values": [1]}"#))
        XCTAssertNil(ChartSpec.decode(#"{"type": "bar", "labels": [], "values": []}"#))
    }

    func testSeriesArePaddedOrTrimmedToLabels() {
        let s = ChartSpec.decode(#"{"type": "line", "labels": ["a", "b", "c"], "series": [{"name": "x", "values": [1, 2, 3, 4]}]}"#)
        XCTAssertEqual(s?.series.first?.values, [1, 2, 3])
        let short = ChartSpec.decode(#"{"type": "line", "labels": ["a", "b", "c"], "values": [1]}"#)
        XCTAssertEqual(short?.series.first?.values.count, 3)
        XCTAssertTrue(short?.series.first?.values[2].isNaN ?? false)
    }

    func testDonutFoldsSmallSlicesIntoOther() {
        let labels = (1...9).map { "\"s\($0)\"" }.joined(separator: ",")
        let values = (1...9).map(String.init).joined(separator: ",")
        let s = ChartSpec.decode("{\"type\": \"donut\", \"labels\": [\(labels)], \"values\": [\(values)]}")
        XCTAssertEqual(s?.labels, ["s9", "s8", "s7", "s6", "s5", "Other"])
        XCTAssertEqual(s?.series.first?.values, [9, 8, 7, 6, 5, 1 + 2 + 3 + 4])
    }

    func testFormat() {
        XCTAssertEqual(ChartSpec.format(42), "42")
        XCTAssertEqual(ChartSpec.format(6.6), "6.6")
        XCTAssertEqual(ChartSpec.format(0.12345), "0.123")
        XCTAssertEqual(ChartSpec.format(123.45), "123.5")
        XCTAssertEqual(ChartSpec.format(.nan), "–")
    }

    func testDataTable() {
        let s = ChartSpec(kind: .bar, xLabel: "Size", labels: ["M5", "M6"],
                          series: [.init(name: "Blind", values: [42, 18]), .init(name: "Through", values: [0, 6.5])])
        XCTAssertEqual(s.table.header, ["Size", "Blind", "Through"])
        XCTAssertEqual(s.table.rows, [["M5", "42", "0"], ["M6", "18", "6.5"]])
        XCTAssertEqual(s.table.alignments, [.leading, .trailing, .trailing])
    }
}

final class ReplyAssemblerTests: XCTestCase {
    func testNotesBeforeToolCallsBecomeWorking() {
        var r = ReplyAssembler()
        r.content("I'll pull the cylindrical faces to get exact diameters.")
        r.toolCall("Searching faces")
        r.content("The search caps at 50, so let me sample.")
        r.toolCall("Inspecting feature")
        r.toolCall("Inspecting feature")
        r.content("## Hole census\n169 holes.")
        r.finish()
        XCTAssertEqual(r.answer, "## Hole census\n169 holes.")
        XCTAssertEqual(r.work, [
            .note("I'll pull the cylindrical faces to get exact diameters."),
            .lookup("Searching faces"),
            .note("The search caps at 50, so let me sample."),
            .lookup("Inspecting feature"),
            .lookup("Inspecting feature"),
        ])
    }

    func testAnswerWrittenBeforeAFinalLookupIsKept() {
        var r = ReplyAssembler()
        r.content("The bore is ∅20 mm.")
        r.toolCall("Measuring distance")
        r.finish()
        XCTAssertEqual(r.answer, "The bore is ∅20 mm.")
        XCTAssertEqual(r.work, [.lookup("Measuring distance")])
    }

    func testShowingInTheViewDoesNotSplitTheAnswer() {
        var r = ReplyAssembler()
        r.content("The thinnest wall is **1.6 mm**, between F12 and F30.")
        r.aside("Showing it in the 3D view")
        r.content("(highlighted in the view)")
        r.finish()
        XCTAssertEqual(r.answer, "The thinnest wall is **1.6 mm**, between F12 and F30.\n\n(highlighted in the view)")
        XCTAssertEqual(r.work, [.lookup("Showing it in the 3D view")])
    }

    func testPlainAnswerHasNoWorking() {
        var r = ReplyAssembler()
        r.content("Hello ")
        r.content("there.")
        r.finish()
        XCTAssertEqual(r.answer, "Hello there.")
        XCTAssertEqual(r.work, [])
    }
}

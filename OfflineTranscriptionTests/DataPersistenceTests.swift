import XCTest
@testable import OfflineTranscription

/// Tests for utility functions (persistence tests removed — TranscriptionRecord not in this target).
@MainActor
final class DataPersistenceTests: XCTestCase {

    // MARK: - Utilities

    func testFormatDurationEdgeCases() {
        XCTAssertEqual(FormatUtils.formatDuration(0), "0:00")
        XCTAssertEqual(FormatUtils.formatDuration(-1), "0:00") // negative clamped
        XCTAssertEqual(FormatUtils.formatDuration(59), "0:59")
        XCTAssertEqual(FormatUtils.formatDuration(60), "1:00")
        XCTAssertEqual(FormatUtils.formatDuration(3599), "59:59")
        XCTAssertEqual(FormatUtils.formatDuration(3600), "1:00:00")
        XCTAssertEqual(FormatUtils.formatDuration(86400), "24:00:00")
    }

    func testFormatFileSizeValues() {
        let small = FormatUtils.formatFileSize(500)
        XCTAssertFalse(small.isEmpty)

        let medium = FormatUtils.formatFileSize(5_000_000)
        XCTAssertTrue(medium.contains("MB") || medium.contains("KB"))

        let large = FormatUtils.formatFileSize(2_000_000_000)
        XCTAssertTrue(large.contains("GB") || large.contains("MB"))
    }
}

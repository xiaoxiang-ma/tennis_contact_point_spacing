// AudioDetectorTests.swift
// TennisContactTests
//
// Unit tests for AudioDetector.
// All tests use synthetic PCM buffers — no video file required.
// Run with Cmd+U in Xcode.
//
// Validation criterion (docs/implementation_v3.md §8.2):
//   Detected timestamps must agree with planted burst positions within ±50 ms.

import XCTest
@testable import TennisContact

final class AudioDetectorTests: XCTestCase {

    private let detector = AudioDetector()
    private let sampleRate = 44_100.0

    // MARK: - Helpers

    /// Synthesise a mono PCM buffer containing one or more short tone bursts
    /// at known timestamps, with a quiet low-frequency noise floor that sits
    /// below the bandpass (100 Hz) so it is removed by the filter.
    private func makeSyntheticBuffer(
        durationSeconds: Double,
        burstTimestamps: [Double],   // seconds from start
        burstFrequency: Double = 2_000,  // Hz — mid-band, reliably detected
        burstDurationMs: Double = 15     // ms — realistic ball impact length
    ) -> [Float] {
        let totalSamples = Int(sampleRate * durationSeconds)
        var signal = [Float](repeating: 0, count: totalSamples)

        // Low-frequency noise floor (100 Hz, amplitude 0.01) — outside the
        // 1–4 kHz band so the filter removes it before threshold computation.
        for i in 0..<totalSamples {
            let phase = 2.0 * Double.pi * 100.0 * Double(i) / sampleRate
            signal[i] = Float(sin(phase)) * 0.01
        }

        // Tone bursts at each planted timestamp
        let burstSamples = Int(sampleRate * burstDurationMs / 1_000.0)
        for t in burstTimestamps {
            let start = Int(t * sampleRate)
            guard start + burstSamples < totalSamples else { continue }
            for j in 0..<burstSamples {
                let phase = 2.0 * Double.pi * burstFrequency * Double(j) / sampleRate
                signal[start + j] += Float(sin(phase)) * 0.8
            }
        }

        return signal
    }

    // MARK: - Core Detection Tests

    /// A single tone burst at a known timestamp must be detected within ±50 ms.
    func testDetectsSingleBurstWithinTolerance() {
        let planted = 1.5   // seconds
        let signal = makeSyntheticBuffer(durationSeconds: 5.0, burstTimestamps: [planted])

        let contacts = detector.detectInBuffer(signal, sampleRate: sampleRate)

        XCTAssertEqual(contacts.count, 1, "Expected exactly one contact for a single burst")
        guard let contact = contacts.first else { return }

        XCTAssertEqual(
            contact.timestamp, planted,
            accuracy: 0.050,
            "Detected timestamp \(contact.timestamp)s should be within ±50 ms of planted \(planted)s"
        )
    }

    /// Two bursts separated by 1 second (> 300 ms min-gap) must both be detected.
    func testDetectsMultipleBurstsWithMinGap() {
        let planted = [1.0, 2.5]
        let signal = makeSyntheticBuffer(durationSeconds: 5.0, burstTimestamps: planted)

        let contacts = detector.detectInBuffer(signal, sampleRate: sampleRate)

        XCTAssertEqual(contacts.count, 2, "Both bursts should be detected (gap > minGapMs)")
        for (contact, expected) in zip(contacts, planted) {
            XCTAssertEqual(
                contact.timestamp, expected,
                accuracy: 0.050,
                "Contact at \(contact.timestamp)s should match planted \(expected)s within ±50 ms"
            )
        }
    }

    /// Two bursts 100 ms apart (< 300 ms min-gap) should collapse to one contact.
    func testNMSCollapsesBurstsWithinMinGap() {
        let planted = [1.0, 1.1]   // 100 ms apart < 300 ms min-gap
        let signal = makeSyntheticBuffer(durationSeconds: 5.0, burstTimestamps: planted)

        let contacts = detector.detectInBuffer(signal, sampleRate: sampleRate)

        XCTAssertEqual(
            contacts.count, 1,
            "Two bursts within minGapMs should be merged into one contact by NMS"
        )
    }

    /// Silent audio (all zeros) must return no contacts.
    func testSilentAudioReturnsNoContacts() {
        let silence = [Float](repeating: 0, count: Int(sampleRate * 5.0))
        let contacts = detector.detectInBuffer(silence, sampleRate: sampleRate)
        XCTAssertTrue(contacts.isEmpty, "Silent audio should produce no contacts")
    }

    /// Empty buffer must return no contacts without crashing.
    func testEmptyBufferReturnsNoContacts() {
        let contacts = detector.detectInBuffer([], sampleRate: sampleRate)
        XCTAssertTrue(contacts.isEmpty, "Empty buffer should produce no contacts")
    }

    // MARK: - Scoring Tests

    /// A short, symmetric peak must outscore a loud, wide, asymmetric one.
    /// This is the core property that lets ball impacts beat shoe screeches.
    func testImpactScoreFavorsNarrowSymmetricPeaks() {
        // Good: quiet but short and symmetric (real ball impact)
        let goodScore = detector.impactScore(
            height: 0.3, maxHeight: 1.0,
            fwhmMs: 10,  symmetry: 0.95
        )
        // Bad: loud but wide and asymmetric (shoe screech)
        let badScore = detector.impactScore(
            height: 1.0, maxHeight: 1.0,
            fwhmMs: 80,  symmetry: 0.2
        )
        XCTAssertGreaterThan(
            goodScore, badScore,
            "Short symmetric peak (\(goodScore)) should outscore loud asymmetric peak (\(badScore))"
        )
    }

    /// At ideal FWHM (~12 ms = maxImpactFwhmMs × 0.3), narrowness score should be 1.0.
    func testNarrowScoreIsMaxAtIdealFWHM() {
        let idealFwhm = detector.maxImpactFwhmMs * 0.3   // 12 ms by default
        let score = detector.impactScore(
            height: 1.0, maxHeight: 1.0,
            fwhmMs: idealFwhm, symmetry: 1.0
        )
        // Max possible: 0.2*1 + 0.4*1 + 0.4*1 = 1.0
        XCTAssertEqual(score, 1.0, accuracy: 0.001)
    }

    // MARK: - Peak Shape Tests

    /// A symmetric triangular envelope peak should return symmetry ≈ 1.0
    /// and a FWHM matching the geometry of the triangle.
    func testMeasurePeakShapeSymmetricTriangle() {
        // Build a triangular peak: rises from 0 to 1 over 220 samples,
        // falls back to 0 over 220 samples. Total width 440 samples.
        // Half-max (0.5) is reached at 110 samples each side → FWHM = 220 samples.
        // At 44100 Hz: FWHM = 220/44100 * 1000 ≈ 4.99 ms
        let halfWidth = 220
        let peakIdx   = halfWidth
        let n         = halfWidth * 2 + 1
        var envelope  = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let dist = abs(i - peakIdx)
            envelope[i] = max(0, Float(halfWidth - dist)) / Float(halfWidth)
        }

        let (fwhmMs, symmetry) = detector.measurePeakShape(
            envelope, peakIndex: peakIdx, sampleRate: sampleRate
        )

        let expectedFwhmMs = Float(halfWidth) * 1000.0 / Float(sampleRate)  // ~4.99 ms
        XCTAssertEqual(fwhmMs, expectedFwhmMs, accuracy: 0.5,
                       "FWHM should match triangle geometry (expected ~\(expectedFwhmMs) ms)")
        XCTAssertEqual(symmetry, 1.0, accuracy: 0.01,
                       "Symmetric triangle should have symmetry ≈ 1.0")
    }

    /// An asymmetric peak (fast rise, slow fall) should have symmetry < 0.5.
    func testMeasurePeakShapeAsymmetric() {
        // Fast rise: 10 samples. Slow fall: 100 samples.
        let rise  = 10
        let fall  = 100
        let total = rise + fall + 1
        var envelope = [Float](repeating: 0, count: total)
        let peak = rise
        for i in 0...rise { envelope[i] = Float(i) / Float(rise) }
        for i in 0...fall { envelope[peak + i] = Float(fall - i) / Float(fall) }

        let (_, symmetry) = detector.measurePeakShape(
            envelope, peakIndex: peak, sampleRate: sampleRate
        )

        XCTAssertLessThan(symmetry, 0.5, "Fast-rise/slow-fall peak should have symmetry < 0.5")
    }

    // MARK: - Confidence Tests

    /// The highest-scoring contact should have confidence = 1.0.
    func testTopContactHasConfidence1() {
        let signal  = makeSyntheticBuffer(durationSeconds: 5.0, burstTimestamps: [2.0])
        let contacts = detector.detectInBuffer(signal, sampleRate: sampleRate)

        guard let top = contacts.max(by: { $0.confidence < $1.confidence }) else {
            return XCTFail("No contacts detected")
        }
        XCTAssertEqual(top.confidence, 1.0, accuracy: 0.001,
                       "The best contact should have confidence 1.0")
    }

    /// All confidence values must be in [0.4, 1.0] as specified by the Python formula.
    func testConfidenceValuesInExpectedRange() {
        let signal   = makeSyntheticBuffer(
            durationSeconds: 10.0,
            burstTimestamps: [1.0, 3.0, 5.0, 7.0]
        )
        let contacts = detector.detectInBuffer(signal, sampleRate: sampleRate)

        for c in contacts {
            XCTAssertGreaterThanOrEqual(c.confidence, 0.4,
                "Confidence floor is 0.4 (matches Python)")
            XCTAssertLessThanOrEqual(c.confidence, 1.0,
                "Confidence ceiling is 1.0")
        }
    }
}

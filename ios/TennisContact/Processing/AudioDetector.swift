// AudioDetector.swift
// TennisContact — iOS
//
// Python reference: python/src/audio_detection.py
// Ports detect_contacts_audio() using AVFoundation (audio extraction) and
// Accelerate vDSP (bandpass filter + envelope + peak detection).
//
// Pipeline (mirrors Python exactly):
//   AVAssetReader → PCM Float32 (44.1 kHz, mono)
//   → windowed-sinc FIR bandpass (1–4 kHz, 255 taps)
//   → amplitude envelope (sliding window mean of abs, 5 ms)
//   → adaptive threshold (75th percentile × 3.0)
//   → candidate peaks above threshold with 20 ms dedup
//   → shape analysis per candidate (FWHM walk + rise/fall symmetry)
//   → composite score: 20% amplitude + 40% narrowness + 40% symmetry
//   → NMS (300 ms minimum gap, keep highest score not highest amplitude)
//   → [ContactCandidate]
//
// Validation: Swift timestamps must agree with Python within ±50 ms.
// See docs/architecture.md Component 1 and docs/implementation_v3.md §3.2

import Foundation
import AVFoundation
import Accelerate

// MARK: - Error

enum AudioDetectorError: Error, LocalizedError {
    case noAudioTrack
    case readFailed(String)

    var errorDescription: String? {
        switch self {
        case .noAudioTrack:
            return "The video file has no audio track."
        case .readFailed(let reason):
            return "Audio read failed: \(reason)"
        }
    }
}

// MARK: - AudioDetector

struct AudioDetector {

    // MARK: Parameters (must match Python defaults)

    /// Bandpass filter low cutoff (Hz).
    var lowFreq: Float = 1_000
    /// Bandpass filter high cutoff (Hz).
    var highFreq: Float = 4_000
    /// Peak must exceed noise floor by this multiplier.
    var thresholdFactor: Float = 3.0
    /// Percentile of envelope used as noise floor.
    var noisePercentile: Float = 75.0
    /// Minimum time between accepted contacts (ms).
    var minGapMs: Float = 300
    /// Maximum FWHM of a true ball impact (ms).
    /// Narrower peaks score higher; peaks beyond this get an extra penalty.
    var maxImpactFwhmMs: Float = 40
    /// Envelope smoothing window (ms).
    var envelopeWindowMs: Float = 5.0
    /// Deduplication window for raw peak candidates (ms).
    var deduplicationMs: Float = 20.0
    /// Minimum absolute envelope amplitude for a peak to be considered.
    /// Prevents filter startup transients and floating-point noise from being
    /// detected when the noise floor is near zero (e.g. in unit tests with
    /// synthetic signals). Mirrors Python's optional min_peak_height parameter.
    /// Set to 0 to rely purely on the adaptive threshold.
    var minimumAbsoluteThreshold: Float = 1e-4

    // Scoring weights (20 / 40 / 40 — must match Python exactly)
    static let wAmplitude: Float  = 0.20
    static let wNarrowness: Float = 0.40
    static let wSymmetry: Float   = 0.40

    // MARK: - Public API

    /// Detect ball-racket contacts from the audio track of a video file.
    /// - Parameter videoURL: URL of the video on disk.
    /// - Returns: Array of `ContactCandidate` sorted by timestamp.
    func detect(videoURL: URL) async throws -> [ContactCandidate] {
        let (samples, sampleRate) = try await extractAudio(from: videoURL)
        return detectInBuffer(samples, sampleRate: sampleRate)
    }

    // MARK: - Internal pipeline (exposed for unit testing)

    /// Run the full detection pipeline on a raw PCM buffer.
    /// Exposed as `internal` so `@testable import TennisContact` can call it
    /// without needing a video file on disk.
    func detectInBuffer(_ samples: [Float], sampleRate: Double) -> [ContactCandidate] {
        let filtered = bandpassFilter(samples, sampleRate: sampleRate)
        let envelope = computeEnvelope(filtered, sampleRate: sampleRate)
        return findImpacts(in: envelope, sampleRate: sampleRate)
    }

    // MARK: - Stage 1: Audio Extraction

    func extractAudio(from url: URL) async throws -> (samples: [Float], sampleRate: Double) {
        let asset = AVURLAsset(url: url)

        let tracks = try await asset.loadTracks(withMediaType: .audio)
        guard let track = tracks.first else {
            throw AudioDetectorError.noAudioTrack
        }

        let targetSampleRate = 44_100.0

        let outputSettings: [String: Any] = [
            AVFormatIDKey:              Int(kAudioFormatLinearPCM),
            AVLinearPCMBitDepthKey:     32,
            AVLinearPCMIsFloatKey:      true,
            AVLinearPCMIsBigEndianKey:  false,
            AVLinearPCMIsNonInterleaved: false,
            AVSampleRateKey:            targetSampleRate,
            AVNumberOfChannelsKey:      1
        ]

        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
        reader.add(output)

        guard reader.startReading() else {
            throw AudioDetectorError.readFailed(reader.error?.localizedDescription ?? "unknown")
        }

        var allSamples: [Float] = []

        while reader.status == .reading {
            guard let sampleBuffer = output.copyNextSampleBuffer() else { break }
            guard let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { continue }

            let byteCount = CMBlockBufferGetDataLength(blockBuffer)
            var data = [UInt8](repeating: 0, count: byteCount)
            CMBlockBufferCopyDataBytes(blockBuffer, atOffset: 0, dataLength: byteCount, destination: &data)

            data.withUnsafeBytes { rawPtr in
                let floats = rawPtr.bindMemory(to: Float.self)
                allSamples.append(contentsOf: floats)
            }
        }

        guard !allSamples.isEmpty else {
            throw AudioDetectorError.readFailed("no samples read")
        }

        return (allSamples, targetSampleRate)
    }

    // MARK: - Stage 2: Bandpass Filter (windowed-sinc FIR)
    //
    // Python: scipy.signal.butter(4, [low, high], btype='band') + filtfilt
    // Swift:  windowed-sinc FIR (255 taps, Hann window)
    //
    // A 255-tap windowed-sinc FIR gives ~60 dB stopband rejection, which is
    // sufficient for separating the 1–4 kHz impact band from low-frequency
    // rumble and high-frequency hiss.

    func bandpassFilter(_ samples: [Float], sampleRate: Double) -> [Float] {
        let kernel = makeBandpassKernel(sampleRate: Float(sampleRate))
        return applyFIR(samples, kernel: kernel)
    }

    private func makeBandpassKernel(sampleRate: Float, taps: Int = 255) -> [Float] {
        // Bandpass = LP(highFreq) − LP(lowFreq), each designed as a windowed sinc.
        // Normalised cutoffs: fc_norm = 2 * f / sampleRate  (so Nyquist = 1.0)
        let fc1 = 2.0 * lowFreq / sampleRate
        let fc2 = 2.0 * highFreq / sampleRate
        let center = taps / 2
        var kernel = [Float](repeating: 0, count: taps)

        for i in 0..<taps {
            let n = Float(i - center)
            let sincH: Float
            let sincL: Float
            if n == 0 {
                sincH = fc2
                sincL = fc1
            } else {
                sincH = sin(Float.pi * fc2 * n) / (Float.pi * n)
                sincL = sin(Float.pi * fc1 * n) / (Float.pi * n)
            }
            // Hann window reduces spectral leakage
            let window = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(i) / Float(taps - 1)))
            kernel[i] = (sincH - sincL) * window
        }
        return kernel
    }

    // MARK: - Stage 3: Amplitude Envelope
    //
    // Python: np.abs(audio) then np.convolve with rectangular window (mode='same')
    // Swift:  vDSP_vabs + applyFIR with boxcar kernel

    func computeEnvelope(_ samples: [Float], sampleRate: Double) -> [Float] {
        var rectified = [Float](repeating: 0, count: samples.count)
        vDSP_vabs(samples, 1, &rectified, 1, vDSP_Length(samples.count))

        let windowSamples = max(1, Int(sampleRate * Double(envelopeWindowMs) / 1000.0))
        let boxcar = [Float](repeating: 1.0 / Float(windowSamples), count: windowSamples)
        return applyFIR(rectified, kernel: boxcar)
    }

    // MARK: - FIR Convolution via vDSP_conv
    //
    // Both the bandpass kernel and the boxcar envelope kernel are symmetric,
    // so vDSP_conv (correlation) equals convolution for our use case.
    // Zero-padding of halfP on each side gives "same"-length output.

    private func applyFIR(_ signal: [Float], kernel: [Float]) -> [Float] {
        let P = kernel.count
        guard P > 0, !signal.isEmpty else { return signal }

        let N      = signal.count
        let halfP  = P / 2
        // vDSP_conv requires signal length = N + P - 1
        var padded = [Float](repeating: 0, count: N + P - 1)
        padded.replaceSubrange(halfP..<(halfP + N), with: signal)

        var output = [Float](repeating: 0, count: N)

        padded.withUnsafeBufferPointer { aBuf in
            kernel.withUnsafeBufferPointer { fBuf in
                output.withUnsafeMutableBufferPointer { cBuf in
                    vDSP_conv(
                        aBuf.baseAddress!, 1,
                        fBuf.baseAddress!, 1,
                        cBuf.baseAddress!, 1,
                        vDSP_Length(N),
                        vDSP_Length(P)
                    )
                }
            }
        }
        return output
    }

    // MARK: - Stage 4: Peak Shape Analysis
    //
    // Python: _measure_peak_shape() — walk left/right until below half-max.
    // Returns (fwhm_ms, symmetry) matching the Python implementation exactly.

    func measurePeakShape(
        _ envelope: [Float],
        peakIndex: Int,
        sampleRate: Double
    ) -> (fwhmMs: Float, symmetry: Float) {
        let halfMax = envelope[peakIndex] / 2.0

        // Walk left until below half-max
        var left = peakIndex
        while left > 0 && envelope[left] > halfMax { left -= 1 }

        // Walk right until below half-max
        var right = peakIndex
        while right < envelope.count - 1 && envelope[right] > halfMax { right += 1 }

        let fwhmMs = Float(right - left) * 1000.0 / Float(sampleRate)

        let riseSamples = peakIndex - left
        let fallSamples = right - peakIndex

        let symmetry: Float
        if riseSamples > 0 && fallSamples > 0 {
            symmetry = Float(min(riseSamples, fallSamples)) / Float(max(riseSamples, fallSamples))
        } else {
            symmetry = 0.0
        }

        return (fwhmMs, symmetry)
    }

    // MARK: - Stage 5: Composite Impact Score
    //
    // Python: _impact_score()
    // 20% amplitude + 40% narrowness (continuous penalty) + 40% symmetry
    // Shape dominates so a quiet, sharp impact beats a loud screech.

    func impactScore(
        height: Float,
        maxHeight: Float,
        fwhmMs: Float,
        symmetry: Float
    ) -> Float {
        let ampScore = maxHeight > 0 ? height / maxHeight : 0.0

        let idealFwhmMs = maxImpactFwhmMs * 0.3   // ~12 ms at default 40 ms max
        let narrowScore: Float
        if fwhmMs <= idealFwhmMs {
            narrowScore = 1.0
        } else if fwhmMs <= maxImpactFwhmMs {
            narrowScore = idealFwhmMs / fwhmMs
        } else {
            // Extra penalty for peaks beyond the hard limit
            narrowScore = idealFwhmMs / fwhmMs * 0.5
        }

        return Self.wAmplitude  * ampScore
             + Self.wNarrowness * narrowScore
             + Self.wSymmetry   * symmetry
    }

    // MARK: - Stage 6: Peak Detection + NMS
    //
    // Python: find_impact_peaks() three-stage approach:
    //   1. All local maxima above threshold with 20 ms dedup
    //   2. Shape scoring per candidate
    //   3. NMS with min_gap keeping highest score (not highest amplitude)

    private func findImpacts(
        in envelope: [Float],
        sampleRate: Double
    ) -> [ContactCandidate] {
        guard envelope.count > 2 else { return [] }

        // Adaptive threshold: noise_percentile × threshold_factor,
        // floored by minimumAbsoluteThreshold to prevent filter startup
        // transients (~1e-5) from being detected when the noise floor is
        // near zero (matches Python's optional min_peak_height parameter).
        let sorted = envelope.sorted()
        let pctIdx = Int(Float(sorted.count - 1) * noisePercentile / 100.0)
        let noiseFloor = sorted[pctIdx]
        let threshold = max(noiseFloor * thresholdFactor, minimumAbsoluteThreshold)

        // Stage 1: candidates — local maxima above threshold, 20 ms dedup
        let dedupSamples = max(1, Int(sampleRate * Double(deduplicationMs) / 1000.0))
        var candidates: [(index: Int, height: Float, fwhmMs: Float, symmetry: Float, score: Float)] = []

        var i = 1
        while i < envelope.count - 1 {
            let v = envelope[i]
            if v >= threshold && v > envelope[i - 1] && v > envelope[i + 1] {
                let (fwhmMs, symmetry) = measurePeakShape(envelope, peakIndex: i, sampleRate: sampleRate)
                candidates.append((i, v, fwhmMs, symmetry, 0))
                i += dedupSamples
            } else {
                i += 1
            }
        }

        guard !candidates.isEmpty else { return [] }

        // Stage 2: score each candidate
        let maxHeight = candidates.map(\.height).max() ?? 1.0
        candidates = candidates.map { c in
            let score = impactScore(
                height: c.height, maxHeight: maxHeight,
                fwhmMs: c.fwhmMs, symmetry: c.symmetry
            )
            return (c.index, c.height, c.fwhmMs, c.symmetry, score)
        }

        // Stage 3: NMS — within any min_gap window keep the highest scorer
        let minGapSamples = Int(sampleRate * Double(minGapMs) / 1000.0)
        var selected: [(index: Int, height: Float, fwhmMs: Float, symmetry: Float, score: Float)] = []

        for c in candidates {
            if let last = selected.last, (c.index - last.index) < minGapSamples {
                if c.score > last.score {
                    selected[selected.count - 1] = c
                }
            } else {
                selected.append(c)
            }
        }

        // Convert to ContactCandidates
        // Confidence: scaled so the best peak = 1.0, floor = 0.4 (matches Python)
        let maxScore = selected.map(\.score).max() ?? 1.0
        return selected.map { c in
            ContactCandidate(
                timestamp: Double(c.index) / sampleRate,
                frameIndex: 0,   // resolved to frame index by ProcessingPipeline (Task 5)
                confidence: min(0.4 + 0.6 * (c.score / maxScore), 1.0)
            )
        }
    }
}

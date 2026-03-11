// OnboardingView.swift
// TennisContact — iOS
//
// Dominant hand selection and camera setup guide (shown once at first launch).
// Stores dominant hand choice in UserDefaults key "dominantSide".
// Sets UserDefaults["hasCompletedOnboarding"] = true on completion.
// See docs/implementation_v3.md Section 2.1

import SwiftUI

struct OnboardingView: View {
    @Binding var isPresented: Bool
    @AppStorage("dominantSide") private var dominantSideRaw: String = DominantSide.right.rawValue
    @State private var page = 0

    var body: some View {
        TabView(selection: $page) {
            handPickerPage.tag(0)
            cameraGuidePage.tag(1)
        }
        .tabViewStyle(.page(indexDisplayMode: .always))
        .indexViewStyle(.page(backgroundDisplayMode: .always))
    }

    // MARK: - Page 1: dominant hand picker

    private var handPickerPage: some View {
        VStack(spacing: 32) {
            Spacer()

            VStack(spacing: 12) {
                Text("Welcome to TennisContact")
                    .font(.title.bold())
                    .multilineTextAlignment(.center)
                Text("Which hand do you play with?")
                    .font(.title3)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            .padding(.horizontal, 24)

            HStack(spacing: 20) {
                HandCard(
                    label: "Right",
                    systemImage: "hand.point.right.fill",
                    isSelected: dominantSideRaw == DominantSide.right.rawValue
                ) {
                    dominantSideRaw = DominantSide.right.rawValue
                }
                HandCard(
                    label: "Left",
                    systemImage: "hand.point.left.fill",
                    isSelected: dominantSideRaw == DominantSide.left.rawValue
                ) {
                    dominantSideRaw = DominantSide.left.rawValue
                }
            }
            .padding(.horizontal, 32)

            Button("Next") { withAnimation { page = 1 } }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)

            Spacer()
        }
    }

    // MARK: - Page 2: camera setup guide

    private var cameraGuidePage: some View {
        VStack(spacing: 0) {
            Spacer()

            VStack(spacing: 12) {
                Image(systemName: "video.fill")
                    .font(.system(size: 48))
                    .foregroundStyle(.tint)
                Text("Recording Setup")
                    .font(.title.bold())
                Text("For the best results, follow these tips:")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            .padding(.horizontal, 24)
            .padding(.bottom, 32)

            VStack(alignment: .leading, spacing: 20) {
                GuideRow(icon: "mappin.and.ellipse",
                         title: "Tripod behind the baseline",
                         detail: "Position behind and slightly to the side of the player.")
                GuideRow(icon: "ruler",
                         title: "Camera height: 4–6 ft",
                         detail: "Keep the lens roughly at hip-to-shoulder height.")
                GuideRow(icon: "person.fill",
                         title: "Player fills 40–60% of frame",
                         detail: "Full body must be visible throughout each swing.")
                GuideRow(icon: "speedometer",
                         title: "60 fps for best accuracy",
                         detail: "Shoot in 60fps or higher; 30fps is also supported.")
            }
            .padding(.horizontal, 32)

            Spacer()

            Button("Get Started") {
                UserDefaults.standard.set(true, forKey: "hasCompletedOnboarding")
                isPresented = false
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .padding(.bottom, 48)
        }
    }
}

// MARK: - HandCard

private struct HandCard: View {
    let label: String
    let systemImage: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 14) {
                Image(systemName: systemImage)
                    .font(.system(size: 44))
                    .foregroundStyle(isSelected ? Color.accentColor : Color.primary)
                Text(label)
                    .font(.headline)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 28)
            .background(
                isSelected
                    ? Color.accentColor.opacity(0.12)
                    : Color(.secondarySystemBackground)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(isSelected ? Color.accentColor : Color.clear, lineWidth: 2)
            )
            .clipShape(RoundedRectangle(cornerRadius: 16))
        }
        .buttonStyle(.plain)
        .animation(.easeInOut(duration: 0.15), value: isSelected)
    }
}

// MARK: - GuideRow

private struct GuideRow: View {
    let icon: String
    let title: String
    let detail: String

    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            Image(systemName: icon)
                .font(.body)
                .foregroundStyle(.tint)
                .frame(width: 22)
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.subheadline.weight(.semibold))
                Text(detail).font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}

// MARK: - Preview

#Preview {
    OnboardingView(isPresented: .constant(true))
}

// ContentView.swift
// TennisContact — iOS
//
// Root navigation container.
// Shows "Tennis Contact" title and app version on launch.
// See docs/implementation_v3.md Section 4.1

import SwiftUI

struct ContentView: View {
    private let version = Bundle.main
        .infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0"

    var body: some View {
        NavigationStack {
            VStack(spacing: 12) {
                Text("Tennis Contact")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                Text("Version \(version)")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            .navigationTitle("Tennis Contact")
        }
    }
}

#Preview {
    ContentView()
}

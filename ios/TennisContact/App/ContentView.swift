// ContentView.swift
// TennisContact — iOS
//
// Root navigation container. Hosts HomeView inside the app-level NavigationStack.
// Presents OnboardingView as a full-screen sheet on first launch.
// See docs/implementation_v3.md Section 4.1

import SwiftUI

struct ContentView: View {
    @AppStorage("hasCompletedOnboarding") private var hasCompletedOnboarding = false
    @State private var showOnboarding = false

    var body: some View {
        NavigationStack {
            HomeView()
        }
        .sheet(isPresented: $showOnboarding) {
            OnboardingView(isPresented: $showOnboarding)
                .interactiveDismissDisabled()
        }
        .onAppear {
            if !hasCompletedOnboarding {
                showOnboarding = true
            }
        }
    }
}

#Preview {
    ContentView()
}

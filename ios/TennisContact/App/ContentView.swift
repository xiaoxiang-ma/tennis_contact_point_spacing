// ContentView.swift
// TennisContact — iOS
//
// Root navigation container. Hosts HomeView inside the app-level NavigationStack.
// See docs/implementation_v3.md Section 4.1

import SwiftUI

struct ContentView: View {
    var body: some View {
        NavigationStack {
            HomeView()
        }
    }
}

#Preview {
    ContentView()
}

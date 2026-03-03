// HomeView.swift
// TennisContact — iOS
//
// Session list screen. "+" toolbar button opens SessionView as a sheet.
// Session list is populated in Task 12 (CoreData persistence).
// See docs/implementation_v3.md Section 2.2

import SwiftUI

struct HomeView: View {
    @State private var showingNewSession = false

    var body: some View {
        Group {
            // Placeholder until Task 12 adds CoreData session list
            ContentUnavailableView {
                Label("No Sessions Yet", systemImage: "figure.tennis")
            } description: {
                Text("Tap + to analyse your first video.")
            }
        }
        .navigationTitle("Tennis Contact")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    showingNewSession = true
                } label: {
                    Image(systemName: "plus")
                }
            }
        }
        .sheet(isPresented: $showingNewSession) {
            SessionView()
        }
    }
}

#Preview {
    NavigationStack {
        HomeView()
    }
}

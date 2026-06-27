//
//  SSVEP2048App.swift
//  SSVEP2048
//
//  Application entry point (SwiftUI lifecycle). Owns the single `AppViewModel`
//  for the whole process, injects it into the environment, and ties the EEG
//  pipeline's lifetime to the window scene's visibility.
//
//  Naming (contract §B.19): the @main type is `SSVEP2048App`, in this file
//  `App/SSVEP2048App.swift`. There is exactly one `@main` in the target.
//

import SwiftUI

/// The SSVEP-2048 application.
///
/// A single `AppViewModel` is created as a `@StateObject` (owned for the app's
/// lifetime) and published into the environment so every view — most importantly
/// `ContentView` and the `SSVEPController` it observes — shares one pipeline.
///
/// The pipeline is started when the root scene appears and stopped when it
/// disappears, so the EEG source (synthetic timer or live BLE link) and the
/// decode loop only run while the UI is on screen.
@main
struct SSVEP2048App: App {

    /// The one and only view model / pipeline coordinator for the process.
    @StateObject private var viewModel = AppViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(viewModel)
                .onAppear { viewModel.start() }
                .onDisappear { viewModel.stop() }
        }
    }
}

"""run.py — one-command launcher for the whole EEG hackathon kit.

    python run.py            # interactive menu
    python run.py <n>        # run option n directly

Everything degrades gracefully: no headset -> synthetic; no calibration -> FBCCA.
"""
import subprocess, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))

OPTIONS = [
    ("Gaming: SSVEP -> 2048 (FLAGSHIP, self-contained game)", "app/ssvep_2048_app.py", []),
    ("Gaming: SSVEP -> 2048 LIVE (Unicorn LSL)",              "app/ssvep_2048_app.py", ["--live"]),
    ("Gaming: Canabalt FOCUS trigger (band-power, UHB-robust)","canabalt/canabalt_focus.py", []),
    ("Setup: check the 4 SSVEP frequencies on your screen",   "ssvep/ssvep_cca.py", []),
    ("Setup: calibrate TRCA + A/B vs FBCCA (real recording)", "ssvep/ssvep_ab.py", []),
    ("Demo (no HW): SSVEP decoder accuracy/ITR",              "ssvep/ssvep_cca.py", []),
    ("Demo (no HW): motor pipeline end-to-end",               "run_demo.py", []),
    ("Data Analysis: Motor Imagery (CSP+LDA)",                "data_analysis/mi_pipeline.py", []),
    ("Data Analysis: P300 speller (xDAWN+LDA)",               "data_analysis/p300_pipeline.py", []),
]


def main():
    print("\n=== EEG Hackathon Kit (Unicorn Hybrid Black) ===")
    for i, (name, _, _) in enumerate(OPTIONS, 1):
        print(f"  {i}. {name}")
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        choice = int(sys.argv[1])
    else:
        choice = int(input("Select [1-%d]: " % len(OPTIONS)) or "1")
    name, script, args = OPTIONS[choice - 1]
    print(f"\n-> {name}\n")
    subprocess.run([sys.executable, os.path.join(HERE, script), *args], cwd=os.path.dirname(os.path.join(HERE, script)))


if __name__ == "__main__":
    main()

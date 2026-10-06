# Microsoft Band Collection

This folder contains the Microsoft Band collection bridge used for EduGage GSR
and heart-rate streams.

- `BandReceiver.py` runs on the experiment computer. It listens for the UWP app
  on TCP port `9898`, writes `msband_gsr.csv` and `msband_hr.csv`, and exposes a
  local haptic command bridge on `127.0.0.1:9899`.
- `MSBandStreamer/` contains the UWP app source that connects to the paired
  Microsoft Band, streams GSR and heart-rate samples to `BandReceiver.py`, and
  accepts `START`/`STOP` haptic commands.

Build products, packaged installers, Visual Studio cache files, and
`BandUnlocker.exe` are intentionally excluded from this public source snapshot.

Basic run order:

1. Start `BandReceiver.py` on the experiment computer.
2. Build/run `MSBandStreamer` in Visual Studio on the Windows machine paired
   with the Microsoft Band.
3. In the UWP app, enter the experiment computer IP address and port `9898`,
   then connect.

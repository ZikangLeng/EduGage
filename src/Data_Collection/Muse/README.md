# Muse OSC Receiver

`muse_osc.py` records Muse EEG, PPG, accelerometer, gyroscope, and marker
streams relayed over OSC, for example from the Mind Monitor mobile app.

Configure the Muse relay app to send OSC packets to the collection computer on
port `5001`, then run:

```bash
python Data_Collection/Muse/muse_osc.py
```

The script writes one CSV per stream, using the participant/session prefix from
`Data_Collection/config.json`.

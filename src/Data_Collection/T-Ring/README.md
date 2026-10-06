# T-Ring Collection Notes

T-Ring data were collected through the Android-side T-Ring application and
exported as raw `.bin` files. The Android app/APK is not redistributed in this
repository.

The public Figshare dataset includes the raw T-Ring `.bin` exports for sessions
where a ring file was available. Decoding for analysis is implemented in:

```text
src/engagement/ring_parser.py
```

No Python receiver is required in this `Data_Collection/` folder for T-Ring
collection.

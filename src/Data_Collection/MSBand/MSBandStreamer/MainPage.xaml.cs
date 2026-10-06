using Microsoft.Band;
using Microsoft.Band.Sensors;
using Microsoft.Band.Notifications;
using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Windows.UI.Core;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Navigation;
using System.Diagnostics;
using Windows.Networking.Sockets;
using Windows.Storage.Streams;

namespace MSBandStreamer
{
    public sealed partial class MainPage : Page
    {
        private IBandClient _bandClient;
        private StreamSocket _socket;
        private DataWriter _writer;
        private DataReader _reader;
        
        // Prevents GSR and HR threads from writing to the stream simultaneously
        private readonly SemaphoreSlim _writeLock = new SemaphoreSlim(1, 1); 
        
        // Volatile ensures thread-safe reads/writes across the Rx and Buzz loops
        private volatile bool _isBuzzing = false; 

        public MainPage()
        {
            this.InitializeComponent();
            
            // Hook into the actual app termination event
            Application.Current.Suspending += Current_Suspending;
        }

        private async void Current_Suspending(object sender, Windows.ApplicationModel.SuspendingEventArgs e)
        {
            // Request a deferral from Windows to finish teardown before the process is killed
            var deferral = e.SuspendingOperation.GetDeferral();
            await CleanupResourcesAsync();
            deferral.Complete();
        }

        private async void ConnectButton_Click(object sender, RoutedEventArgs e)
        {
            ConnectButton.IsEnabled = false;
            StatusText.Text = "Connecting...";

            string ip = IpAddressInput.Text.Trim();
            string port = PortInput.Text.Trim();

            try
            {
                // 1. Establish Single TCP Connection
                _socket = new StreamSocket();
                await _socket.ConnectAsync(new Windows.Networking.HostName(ip), port);
                
                _writer = new DataWriter(_socket.OutputStream);
                _reader = new DataReader(_socket.InputStream)
                {
                    InputStreamOptions = InputStreamOptions.Partial
                };

                StatusText.Text = "TCP connected. Locating Band...";

                // 2. Connect to Band
                var pairedBands = await BandClientManager.Instance.GetBandsAsync();
                if (pairedBands.Length == 0)
                {
                    StatusText.Text = "No Band paired.";
                    ConnectButton.IsEnabled = true;
                    return;
                }

                _bandClient = await BandClientManager.Instance.ConnectAsync(pairedBands[0]);
                StatusText.Text = "Band connected. Starting streams...";

                // 3. Start Processes
                await StartSensorsAsync();
                _ = ReceiveCommandsLoopAsync(); // Fire and forget Rx loop
            }
            catch (Exception ex)
            {
                StatusText.Text = $"Connection Error: {ex.Message}";
                ConnectButton.IsEnabled = true;
            }
        }

        private async Task StartSensorsAsync()
        {
            // GSR Sensor Setup
            var gsrSensor = _bandClient.SensorManager.Gsr;
            if (gsrSensor.SupportedReportingIntervals.Any())
            {
                gsrSensor.ReportingInterval = gsrSensor.SupportedReportingIntervals.Min(); // Max frequency
            }
            gsrSensor.ReadingChanged += GsrSensor_ReadingChanged;
            await gsrSensor.StartReadingsAsync();

            // Heart Rate Sensor Setup
            var hrSensor = _bandClient.SensorManager.HeartRate;
            if (hrSensor.GetCurrentUserConsent() != UserConsent.Granted)
            {
                await hrSensor.RequestUserConsentAsync();
            }
            if (hrSensor.SupportedReportingIntervals.Any())
            {
                hrSensor.ReportingInterval = hrSensor.SupportedReportingIntervals.Min();
            }
            hrSensor.ReadingChanged += HrSensor_ReadingChanged;
            await hrSensor.StartReadingsAsync();

            StatusText.Text = "Streaming data and listening for commands...";
        }

        // --- SENSOR EVENT HANDLERS ---
        
        private async void GsrSensor_ReadingChanged(object sender, BandSensorReadingEventArgs<IBandGsrReading> e)
        {
            string timestamp = e.SensorReading.Timestamp.ToUnixTimeMilliseconds().ToString();
            string payload = $"GSR,{timestamp},{e.SensorReading.Resistance}\n";
            
            await SendDataAsync(payload);

            _ = Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () =>
            {
                GSRDisplay.Text = $"GSR: {e.SensorReading.Resistance} kOhms";
            });
        }

        private async void HrSensor_ReadingChanged(object sender, BandSensorReadingEventArgs<IBandHeartRateReading> e)
        {
            string timestamp = e.SensorReading.Timestamp.ToUnixTimeMilliseconds().ToString();
            string payload = $"HR,{timestamp},{e.SensorReading.HeartRate},{e.SensorReading.Quality}\n";
            
            await SendDataAsync(payload);

            _ = Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () =>
            {
                HRDisplay.Text = $"HR: {e.SensorReading.HeartRate} bpm";
            });
        }

        // --- NETWORK TX / RX ---

        private async Task SendDataAsync(string message)
        {
            if (_writer == null) return;

            await _writeLock.WaitAsync();
            try
            {
                _writer.WriteString(message);
                await _writer.StoreAsync(); // Flushes to the socket efficiently
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Send Error: {ex.Message}");
            }
            finally
            {
                _writeLock.Release();
            }
        }

        private async Task ReceiveCommandsLoopAsync()
        {
            try
            {
                while (true)
                {
                    await _reader.LoadAsync(512);
                    if (_reader.UnconsumedBufferLength == 0) break; // Server disconnected

                    string command = _reader.ReadString(_reader.UnconsumedBufferLength).Trim().ToUpper();
                    
                    if (command.Contains("START") && !_isBuzzing)
                    {
                        _isBuzzing = true;
                        _ = BuzzLoopAsync();
                    }
                    else if (command.Contains("STOP"))
                    {
                        _isBuzzing = false;
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Rx Loop Error: {ex.Message}");
            }
            finally
            {
                await CleanupResourcesAsync();
            }
        }

        private async Task BuzzLoopAsync()
        {
            while (_isBuzzing && _bandClient != null)
            {
                try
                {
                    await _bandClient.NotificationManager.VibrateAsync(VibrationType.ThreeToneHigh);
                    await Task.Delay(1500); // Prevent crashing the Band's Bluetooth queue
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Buzz Error: {ex.Message}");
                    _isBuzzing = false; 
                }
            }
        }

        // --- LIFECYCLE MANAGEMENT ---

        protected override async void OnNavigatedFrom(NavigationEventArgs e)
        {
            await CleanupResourcesAsync();
            base.OnNavigatedFrom(e);
        }

        private async Task CleanupResourcesAsync()
        {
            _isBuzzing = false;

            if (_bandClient != null)
            {
                try
                {
                    var gsr = _bandClient.SensorManager.Gsr;
                    gsr.ReadingChanged -= GsrSensor_ReadingChanged;
                    await gsr.StopReadingsAsync();

                    var hr = _bandClient.SensorManager.HeartRate;
                    hr.ReadingChanged -= HrSensor_ReadingChanged;
                    await hr.StopReadingsAsync();
                }
                catch { /* Ignore errors during teardown */ }
                
                _bandClient.Dispose();
                _bandClient = null;
            }

            _writer?.Dispose();
            _reader?.Dispose();
            _socket?.Dispose();
            
            _writer = null;
            _reader = null;
            _socket = null;

            await Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () =>
            {
                StatusText.Text = "Disconnected.";
                ConnectButton.IsEnabled = true;
            });
        }
    }
}
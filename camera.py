"""
camera.py
Everything that talks to the FLIR Grasshopper3 (GS3-U3-32S4M) lives here.
The rest of the project only calls:
    open_camera(), grab_frame(), close_camera()
 
Sensor: Sony IMX252, mono, 2048 x 1536, 3.45 um pixels, 12-bit ADC.
"""
 
import PySpin
 
# These are module-level variables so the three functions can share them.
system = None
cam_list = None
cam = None
 
 
def try_set(description, action):
    """Run one camera setting. If this camera doesn't support it, just say so."""
    try:
        action()
        print("  OK   :", description)
    except PySpin.SpinnakerException as error:
        print("  SKIP :", description, "->", error)
 
 
def open_camera(exposure_us=1000.0, gain_db=0.0):
    global system, cam_list, cam
 
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
 
    if cam_list.GetSize() == 0:
        cam_list.Clear()
        system.ReleaseInstance()
        raise RuntimeError("No camera found. Is it plugged into a USB3 port?")
 
    cam = cam_list[0]
    cam.Init()
    print("Connected to:", cam.DeviceModelName.GetValue())
    print("Camera settings:")
 
    # 16-bit pixels so we keep all 12 bits from the sensor.
    try_set("PixelFormat = Mono16",
            lambda: cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono16))
 
    # Manual exposure and gain, so brightness doesn't change by itself
    # (auto modes would change the beam's amplitude and break the profile).
    try_set("ExposureAuto = Off",
            lambda: cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off))
    try_set("ExposureTime = %.0f us" % exposure_us,
            lambda: cam.ExposureTime.SetValue(exposure_us))
    try_set("GainAuto = Off",
            lambda: cam.GainAuto.SetValue(PySpin.GainAuto_Off))
    try_set("Gain = %.1f dB" % gain_db,
            lambda: cam.Gain.SetValue(gain_db))
 
    # Gamma bends the brightness values. A gaussian fit needs them straight
    # (pixel value proportional to light), so turn gamma off.
    try_set("GammaEnable = False",
            lambda: cam.GammaEnable.SetValue(False))
 
    # Always give us the NEWEST frame. Without this, if our fitting is slower
    # than the camera, old frames pile up and the display lags behind.
    try_set("Buffer = NewestOnly",
            lambda: cam.TLStream.StreamBufferHandlingMode.SetValue(
                PySpin.StreamBufferHandlingMode_NewestOnly))
 
    try_set("AcquisitionMode = Continuous",
            lambda: cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous))
 
    cam.BeginAcquisition()
 
 
def grab_frame():
    """Return the latest frame as a numpy array of RAW pixel values, or None."""
    image = cam.GetNextImage(1000)  # wait up to 1000 ms
    if image.IsIncomplete():
        image.Release()
        return None
    raw = image.GetNDArray().copy()
    image.Release()
    return raw
 
 
def close_camera():
    global system, cam_list, cam
    if cam is not None:
        try:
            cam.EndAcquisition()
        except PySpin.SpinnakerException:
            pass
        cam.DeInit()
        del cam          # must delete the camera before releasing the system
        cam = None
    if cam_list is not None:
        cam_list.Clear()
        cam_list = None
    if system is not None:
        system.ReleaseInstance()
        system = None
        
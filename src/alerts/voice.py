import pyttsx3


def speak_warning(message: str):
    """Speaks the given warning message out loud.

    A fresh engine is created (and disposed of) on every call instead of
    sharing one module-level engine. app.py's speak_async() runs this on a
    background thread, and pyttsx3's SAPI5 backend is not safe to drive
    from a thread other than the one that created the engine — reusing a
    single shared engine cross-thread is a known hang/crash hazard on
    Windows.
    """
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate - 25)  # slow down slightly for clarity
    engine.say(message)
    engine.runAndWait()
    engine.stop()

# The Test Block!
if __name__ == "__main__":
    print("Testing voice alert...")
    speak_warning("Warning. Driver distraction detected. Please keep your eyes on the road.")
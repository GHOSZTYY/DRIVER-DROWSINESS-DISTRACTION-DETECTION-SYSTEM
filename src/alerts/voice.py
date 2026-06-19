import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Optional: Slow down the speech rate slightly so it sounds more natural
rate = engine.getProperty('rate')
engine.setProperty('rate', rate - 25)

def speak_warning(message: str):
    """Speaks the given warning message out loud."""
    engine.say(message)
    engine.runAndWait()

# The Test Block!
if __name__ == "__main__":
    print("Testing voice alert...")
    speak_warning("Warning. Driver distraction detected. Please keep your eyes on the road.")
import pygame
import time

# Initialize the audio mixer as soon as the file loads
pygame.mixer.init()

# Load the sound files we just generated into memory
sounds = {
    'low': pygame.mixer.Sound('assets/soft.wav'),
    'medium': pygame.mixer.Sound('assets/medium.wav'),
    'critical': pygame.mixer.Sound('assets/critical.wav')
}

def play_beep(level: str):
    """Plays the corresponding alert sound."""
    if level in sounds:
        sounds[level].play()

# The Test Block!
if __name__ == "__main__":
    print("Testing medium alert...")
    play_beep('medium')
    
    # We need a tiny delay here just for the test, otherwise the script 
    # closes before the sound actually finishes playing!
    time.sleep(2)
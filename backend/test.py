# Simple test script for audio playback
import pygame

pygame.init()
pygame.mixer.init()

choir_file = "sounds/choir1.mp3"
choir_sound = pygame.mixer.Sound(choir_file)
choir_sound.play(-1)

input("Press Enter to stop playback...")
pygame.mixer.quit()


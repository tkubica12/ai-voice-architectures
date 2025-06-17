#!/usr/bin/env python3
"""
Audio test script to diagnose audio playback issues.
"""

import pygame
import numpy as np
import wave
import tempfile
import os
import time

def test_audio_system():
    """Test the audio system with a simple tone."""
    print("🔊 Testing audio system...")
    
    # Initialize pygame mixer
    try:
        pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
        print(f"✅ Pygame mixer initialized: {pygame.mixer.get_init()}")
    except Exception as e:
        print(f"❌ Failed to initialize pygame mixer: {e}")
        return False
    
    # Generate a simple test tone (440 Hz for 2 seconds)
    duration = 2.0
    sample_rate = 22050
    frequency = 440  # A4 note
    
    # Generate sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave_data = np.sin(frequency * 2 * np.pi * t)
    
    # Convert to 16-bit PCM
    audio_data = (wave_data * 32767).astype(np.int16)
    
    # Create a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Write WAV file
        with wave.open(tmp_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        print(f"📁 Created test audio file: {tmp_path}")
        
        # Load and play the sound
        sound = pygame.mixer.Sound(tmp_path)
        print(f"🎵 Playing test tone (440 Hz, {duration}s)...")
        print("🔊 You should hear a beep now!")
        
        channel = sound.play()
        
        # Wait for playback to complete
        while channel.get_busy():
            time.sleep(0.1)
        
        print("✅ Test tone playback completed")
        
    except Exception as e:
        print(f"❌ Error during audio test: {e}")
        return False
    finally:
        # Clean up
        try:
            os.unlink(tmp_path)
        except:
            pass
        pygame.mixer.quit()
    
    return True

def check_audio_devices():
    """Check available audio devices."""
    print("\n🔍 Checking audio devices...")
    
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        
        print(f"Default output device: {p.get_default_output_device_info()['name']}")
        print(f"Default input device: {p.get_default_input_device_info()['name']}")
        
        print("\nAvailable output devices:")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxOutputChannels'] > 0:
                print(f"  {i}: {info['name']} (max channels: {info['maxOutputChannels']})")
        
        p.terminate()
        
    except Exception as e:
        print(f"❌ Error checking audio devices: {e}")

if __name__ == "__main__":
    print("🎯 Audio System Diagnostic Test")
    print("=" * 40)
    
    # Check audio devices
    check_audio_devices()
    
    # Test audio playback
    print("\n" + "=" * 40)
    success = test_audio_system()
    
    if success:
        print("\n✅ Audio test completed successfully!")
        print("If you heard the beep, audio playback is working.")
        print("If you didn't hear anything, check:")
        print("  - System volume levels")
        print("  - Default audio device in Windows")
        print("  - Headphones/speakers are connected")
    else:
        print("\n❌ Audio test failed!")
        print("There may be an issue with your audio configuration.")

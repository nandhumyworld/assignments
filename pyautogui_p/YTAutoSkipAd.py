import pyautogui
import time
import cv2
import numpy as np
#from PIL import Image
import os
import threading
import keyboard

class YouTubeSkipBot:
    def __init__(self, template_paths=None):
        """
        Initialize the YouTube skip button auto-clicker
        
        Args:
            template_paths: List of paths to template images of skip buttons
        """
        self.template_paths = template_paths or ["skip1.png", "skip2.png", "skip3.png"]
        self.templates = self.load_templates()
        self.running = False
        self.confidence_threshold = 0.8
        
        # Set pyautogui settings
        pyautogui.PAUSE = 0.1
        pyautogui.FAILSAFE = True  # Move mouse to top-left corner to stop
        
    def load_templates(self):
        """Load template images for skip button detection"""
        templates = []
        for path in self.template_paths:
            if os.path.exists(path):
                template = cv2.imread(path, cv2.IMREAD_COLOR)
                if template is not None:
                    templates.append(template)
                    print(f"Loaded template: {path}")
                else:
                    print(f"Warning: Could not load template {path}")
            else:
                print(f"Warning: Template file {path} not found")
        
        if not templates:
            print("No template images found. Using fallback text detection.")
        
        return templates
    
    def capture_screen(self):
        """Capture the current screen"""
        screenshot = pyautogui.screenshot()
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    def find_skip_button_by_template(self, screen):
        """Find skip button using template matching"""
        for i, template in enumerate(self.templates):
            # Try multiple scales
            for scale in [0.8, 1.0, 1.2]:
                # Resize template
                height, width = template.shape[:2]
                new_height, new_width = int(height * scale), int(width * scale)
                scaled_template = cv2.resize(template, (new_width, new_height))
                
                # Perform template matching
                result = cv2.matchTemplate(screen, scaled_template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val >= self.confidence_threshold:
                    # Calculate center point
                    center_x = max_loc[0] + new_width // 2
                    center_y = max_loc[1] + new_height // 2
                    print(f"Skip button found using template {i+1} (confidence: {max_val:.2f})")
                    return (center_x, center_y)
        
        return None
    
    def find_skip_button_by_text(self, screen):
        """Fallback method: Find skip button by looking for common skip text patterns"""
        # Convert to grayscale for text detection
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        
        # Common skip button text patterns (you can add more)
        skip_patterns = [
            "Skip Ad",
            "Skip Ads", 
            "Skip",
            "Skip in",
            "광고 건너뛰기",  # Korean
            "Passer",  # French
            "Omitir",  # Spanish
        ]
        
        # This is a basic implementation - for better text detection,
        # you might want to use OCR libraries like pytesseract
        # For now, we'll return None as template matching is preferred
        return None
    
    def click_skip_button(self, position):
        """Click the skip button at the given position"""
        try:
            x, y = position
            
            # Move mouse to position and click
            pyautogui.moveTo(x, y, duration=0.2)
            time.sleep(0.1)
            pyautogui.click()
            
            print(f"Clicked skip button at position ({x}, {y})")
            return True
            
        except Exception as e:
            print(f"Error clicking skip button: {e}")
            return False
    
    def scan_for_skip_button(self):
        """Main scanning loop"""
        print("Starting YouTube skip button scanner...")
        print("Press 'q' to quit, or move mouse to top-left corner")
        
        while self.running:
            try:
                # Capture screen
                screen = self.capture_screen()
                
                # Try to find skip button using templates
                skip_position = None
                if self.templates:
                    skip_position = self.find_skip_button_by_template(screen)
                
                # Fallback to text detection if templates fail
                if skip_position is None:
                    skip_position = self.find_skip_button_by_text(screen)
                
                # Click if found
                if skip_position:
                    if self.click_skip_button(skip_position):
                        print("Successfully clicked skip button!")
                        time.sleep(2)  # Wait a bit after clicking
                
                # Small delay between scans
                time.sleep(0.5)
                
            except pyautogui.FailSafeException:
                print("Failsafe triggered - mouse moved to corner")
                break
            except Exception as e:
                print(f"Error in scan loop: {e}")
                time.sleep(1)
    
    def start(self):
        """Start the skip button detection"""
        if self.running:
            print("Bot is already running!")
            return
        
        self.running = True
        
        # Start scanning in a separate thread
        scan_thread = threading.Thread(target=self.scan_for_skip_button)
        scan_thread.daemon = True
        scan_thread.start()
        
        # Setup keyboard listener for quit
        def on_key_press(event):
            if event.name == 'q':
                self.stop()
        
        keyboard.on_press(on_key_press)
        
        try:
            scan_thread.join()
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the skip button detection"""
        self.running = False
        print("Skip button detection stopped.")

def main():
    """Main function to run the YouTube skip bot"""
    print("YouTube Skip Button Auto-Clicker")
    print("=" * 40)
    
    # Initialize bot with your template images
    template_files = ["skip1.jpg", "skip2.jpg", "skip3.jpg"]
    bot = YouTubeSkipBot(template_files)
    
    print(f"Loaded {len(bot.templates)} template images")
    print("\nInstructions:")
    print("- Make sure your skip button template images are in the same directory")
    print("- Open YouTube videos with ads")
    print("- Press 'q' to quit or move mouse to top-left corner")
    print("- The bot will automatically detect and click skip buttons")
    
    input("\nPress Enter to start the bot...")
    
    try:
        bot.start()
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
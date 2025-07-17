import pyautogui
import keyboard
import time

# Function to find and print the current cursor position
def get_cursor_position():
    print("Move your mouse to the desired position in the next 5 seconds...")
    time.sleep(5)
    x, y = pyautogui.position()
    print(f"Current cursor position: X={x}, Y={y}")
    return (x, y)

# Main function to perform basic automation
def automate_tasks():
    # Wait a few seconds before starting to give you time to switch windows
    time.sleep(3)

    # Type some text
    pyautogui.write("Hello, this is a PyAutoGUI test!", interval=0.05)

    # Press ENTER key
    pyautogui.press('enter')

    # Click at a specific screen position (change as needed)
    # Example: Click at position (500, 500)
    pyautogui.moveTo(500, 500)
    pyautogui.click()
    pyautogui.write("Clicked here!", interval=0.05)

def open_whatsapp_app():
    # Use Keyboard to open the Windows start menu
    keyboard.send('windows')

    #type the name of the app
    pyautogui.write("WhatsApp", interval=0.05)

    # Press ENTER key
    pyautogui.press('enter')
    time.sleep(3)  # Wait for the app to open

if __name__ == "__main__":
    # Get cursor position if you need to know coordinates
    #get_cursor_position()


    # Perform the automated actions
    #automate_tasks()
    open_whatsapp_app()

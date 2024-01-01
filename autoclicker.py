import pyautogui
import time

def auto_clicker(interval, clicks):
    print("Starting Auto Clicker")
    print(f"Clicking {clicks} times with an interval of {interval} seconds")

    for i in range(clicks):
        pyautogui.click()
        time.sleep(interval)

    print("Auto Clicker Finished")

# Example usage: Click 10 times with an interval of 1 second
auto_clicker(1.5, 100)
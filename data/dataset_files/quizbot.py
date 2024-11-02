import tkinter as tk
from PIL import ImageGrab
from utils import process_region, process_image, get_completion, load_api_key
from openai import OpenAI

class QuizBotApp:
    def __init__(self, root):
        self.root = root
        self.start_x, self.start_y, self.temp_region, self.current_selection = 0, 0, None, 0
        self.regions = {i: None for i in range(1, 6)}
        self.client = OpenAI()
        self.client.api_key = load_api_key()
        self.setup_ui()

    def on_button_press(self, event):
        self.start_x, self.start_y = self.root.winfo_pointerx(), self.root.winfo_pointery()

    def on_drag(self, event):
        end_x, end_y = self.root.winfo_pointerx(), self.root.winfo_pointery()
        self.temp_region = (self.start_x, self.start_y, end_x, end_y)

    def on_button_release(self, event):
        self.regions[self.current_selection] = self.temp_region
        self.close_selection_window()
        self.display_text("Region set.\n")

    def close_selection_window(self):
        self.selection_window.quit()
        self.selection_window.destroy()

    def select_window(self, selection_number):
        self.current_selection = selection_number
        selection_type = "Question" if selection_number == 1 else f"Option {selection_number-1}"
        self.display_text(f"Selecting {selection_type}...\nPlease select the region on the screen.\n")
        self.create_selection_window()

    def create_selection_window(self):
        self.selection_window = tk.Toplevel(self.root)
        self.selection_window.attributes("-alpha", 0.3)
        self.selection_window.attributes("-fullscreen", True)
        self.selection_window.bind("<ButtonPress-1>", self.on_button_press)
        self.selection_window.bind("<B1-Motion>", self.on_drag)
        self.selection_window.bind("<ButtonRelease-1>", self.on_button_release)
        self.selection_window.mainloop()

    def display_text(self, text):
        self.text_display.delete("1.0", tk.END)
        self.text_display.insert(tk.END, text)

    def take_screenshots(self):
        ocr_results = []
        for region in self.regions.values():
            if region:
                try:
                    x1, y1, x2, y2 = process_region(region)
                    screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))
                    ocr_results.append(process_image(screenshot))
                except Exception as e:
                    self.display_text(f"Error taking screenshot: {e}\n")
                    return
        prompt = "Question: " + "\n" + "\n".join(ocr_results) + "\nOnly write correct asnwer."
        answer = get_completion(prompt, self.client)
        self.display_text(answer)

    def setup_ui(self):
        self.setup_question_button()
        self.setup_option_buttons()
        self.setup_screenshot_button()
        self.setup_text_display()

    def setup_question_button(self):
        question_button = tk.Button(self.root, text="Question", command=lambda: self.select_window(1))
        question_button.grid(row=0, column=0, columnspan=2)
    
    def setup_option_buttons(self):
        for i in range(2, 6):
            button_text = f"Option {i-1}"
            row = 1 if i < 4 else 2
            column = 0 if i % 2 == 0 else 1
            select_button = tk.Button(self.root, text=button_text, command=lambda i=i: self.select_window(i))
            select_button.grid(row=row, column=column, padx=5)
    
    def setup_screenshot_button(self):
        screenshot_button = tk.Button(self.root, text="Show answer", command=self.take_screenshots)
        screenshot_button.grid(row=3, column=0, columnspan=2)
    
    def setup_text_display(self):
        self.text_display = tk.Text(self.root, height=10, width=50)
        self.text_display.grid(row=4, column=0, columnspan=2)


import pytesseract
from PIL import Image
import openai
import os

def process_region(region):
    x1 = min(region[0], region[2])
    y1 = min(region[1], region[3])
    x2 = max(region[0], region[2])
    y2 = max(region[1], region[3])
    return x1, y1, x2, y2

def process_image(image):
    try:
        gray_image = image.convert('L')
        text = pytesseract.image_to_string(gray_image, lang='eng')
        return text
    except Exception as e:
        print(f"Error in OCR processing: {e}")
        return ""

def get_completion(prompt, client):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error communicating with OpenAI: {e}")
        return ""

def load_api_key():
    return os.getenv("OPENAI_API_KEY", "default_api_key_here")


import tkinter as tk
def main():
    root = tk.Tk()
    app = QuizBotApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

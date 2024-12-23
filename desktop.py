import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import compare4
import numpy as np
import threading
import re

def extract_name(text):
    name = text.strip("'").split('6')[0]  # Gets 'courier'
    return name

def extract_name_regex(text):
    pattern = r"'([a-zA-Z]+)\d"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digital Image Processing Project")
        self.root.geometry("600x400")

        # Label for displaying the image
        self.image_label = tk.Label(self.root, text="Select an image.", font=("Arial", 16))
        self.image_label.pack(pady=20)

        # Label for displaying the detected car name
        self.car_label = tk.Label(self.root, text="", font=("Arial", 16))
        self.car_label.pack(pady=10)

        # Button to select an image
        self.select_button = tk.Button(
            self.root, text="Select Image", command=self.select_image, font=("Arial", 14), bg="lightblue"
        )
        self.select_button.pack(pady=10)

        # Placeholder to store the currently loaded image
        self.loaded_image = None

        # Loading label
        self.loading_label = tk.Label(self.root, text="", font=("Arial", 14), fg="red")
        
    def show_loading(self):
        self.loading_label.pack(pady=10)
        self.loading_label.config(text="Loading, please wait...")

    def hide_loading(self):
        self.loading_label.pack_forget()

    def select_image(self):
        # Open a file dialog to select an image
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All Files", "*.*")]
        )

        if file_path:
            try:
                
                # Process the image after loading
                edge_detected_folder = "EdgeDetectedFolder"
                # Show loading label
                self.show_loading()

                # Run the comparison in a separate thread
                threading.Thread(target=self.process_image, args=(file_path, edge_detected_folder)).start()

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{e}")
        else:
            messagebox.showinfo("No Selection", "Select an image.")

    def process_image(self, input_path, edge_detected_folder):
        results = compare4.compare_with_folder(input_path)
        car_name = results[0]
        car_name = modify_string(car_name)
        self.root.after(0, self.update_car_label, car_name)
        self.root.after(0, self.hide_loading)


    def update_car_label(self, car_name):
        self.car_label.config(text=f"Detected car: {car_name}")

    

def modify_string(text):
   text = str(text)
   text = text[2:]
   dot_index = text.find('.')
   if dot_index != -1:
       text = text[:dot_index-1]
   return text.capitalize()

if __name__ == "__main__":
    # Create the main application window
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
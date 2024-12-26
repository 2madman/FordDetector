import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import compare4
import numpy as np
import threading
import re
from PIL import Image, ImageTk

def extract_name(text):
    name = text.strip("'").split('6')[0]
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
        self.root.geometry("800x600")  # Increased window size

        # Create main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(expand=True, fill='both', padx=10, pady=10)

        # Create frame for images
        self.image_frame = tk.Frame(self.main_frame)
        self.image_frame.pack(fill='both', expand=True)

        # Frame for input image
        self.input_frame = tk.Frame(self.image_frame)
        self.input_frame.pack(side='left', expand=True, padx=5)
        
        # Frame for result image
        self.result_frame = tk.Frame(self.image_frame)
        self.result_frame.pack(side='right', expand=True, padx=5)

        # Labels for images
        self.input_image_label = tk.Label(self.input_frame, text="Input Image", font=("Arial", 12))
        self.input_image_label.pack()
        
        self.result_image_label = tk.Label(self.result_frame, text="Matched Image", font=("Arial", 12))
        self.result_image_label.pack()

        # Image display labels
        self.input_display = tk.Label(self.input_frame)
        self.input_display.pack(pady=5)
        
        self.result_display = tk.Label(self.result_frame)
        self.result_display.pack(pady=5)

        # Control frame for buttons and labels
        self.control_frame = tk.Frame(self.main_frame)
        self.control_frame.pack(side='bottom', fill='x', pady=10)

        # Car name label
        self.car_label = tk.Label(self.control_frame, text="", font=("Arial", 16))
        self.car_label.pack(pady=5)

        # Select button
        self.select_button = tk.Button(
            self.control_frame, 
            text="Select Image", 
            command=self.select_image, 
            font=("Arial", 14), 
            bg="lightblue"
        )
        self.select_button.pack(pady=5)

        # Loading label
        self.loading_label = tk.Label(self.control_frame, text="", font=("Arial", 14), fg="red")
        
        # Store image references to prevent garbage collection
        self.input_photo = None
        self.result_photo = None

    def show_loading(self):
        self.loading_label.pack(pady=5)
        self.loading_label.config(text="Loading, please wait...")

    def hide_loading(self):
        self.loading_label.pack_forget()

    def resize_image(self, image_path, max_size=(300, 300)):
        image = Image.open(image_path)
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(image)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All Files", "*.*")]
        )

        if file_path:
            try:
                # Display input image
                self.input_photo = self.resize_image(file_path)
                self.input_display.config(image=self.input_photo)
                
                # Show loading label
                self.show_loading()

                # Process image in separate thread
                threading.Thread(target=self.process_image, args=(file_path,)).start()

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{e}")
        else:
            messagebox.showinfo("No Selection", "Select an image.")

    def process_image(self, input_path):
        results = compare4.compare_with_folder(input_path)
        car_name = results[0]
        #car_name = modify_string(car_name)
        car_name = delete_until_slash(car_name)

        # Get the path of the matched image
        matched_image_path = f"Cars/{car_name.lower()}"  # Adjust path format as needed
        
        car_name = delete_after_number(car_name)

        self.root.after(0, self.update_display, matched_image_path, car_name.capitalize())
        self.root.after(0, self.hide_loading)

    def update_display(self, matched_image_path, car_name):
        try:
            # Display result image
            self.result_photo = self.resize_image(matched_image_path)
            self.result_display.config(image=self.result_photo)
            
            # Update car label
            self.car_label.config(text=f"Detected car: {car_name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display result image:\n{e}")

def modify_string(text):
    text = str(text)
    text = text[2:]
    dot_index = text.find('.')
    if dot_index != -1:
        text = text[:dot_index-1]
    return text.capitalize()

def delete_until_slash(text):
    return text[0].split('/')[-1]  # Get first element of tuple and get last part after slash

def delete_after_number(text):
    if isinstance(text, tuple):
        text = text[0]
        
    for i, char in enumerate(text):
        if char.isdigit():
            return text[:i].capitalize()
    
    return text.split('.')[0].capitalize()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
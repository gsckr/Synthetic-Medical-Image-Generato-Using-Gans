import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
import os

generator = None
generated_images = []

def load_model():
    global generator
    file_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.h5;*.keras")])
    if file_path:
        try:
            generator = tf.keras.models.load_model(file_path)
            status_label.config(text="Model Loaded Successfully!")
        except Exception as e:
            status_label.config(text=f"Error loading model: {str(e)}")
    else:
        status_label.config(text="No model selected.")

def generate_images():
    global generated_images
    if generator is None:
        status_label.config(text="Load model first!")
        return
    
    noise = np.random.normal(0, 1, (16, 100))  # Generate noise for 4 images
    generated_images = generator.predict(noise)

    generated_images = (generated_images * 127.5 + 127.5).astype(np.uint8)
    
    image_list = []
    for i in range(16):
        img = generated_images[i].squeeze()
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img = Image.fromarray(img)
        image_list.append(img)
    
    display_images(image_list)

def display_images(images):
    for widget in image_frame.winfo_children():
        widget.destroy()
    
    for i, img in enumerate(images):
        img_resized = img.resize((128, 128))  
        img_tk = ImageTk.PhotoImage(img_resized)
        lbl = tk.Label(image_frame, image=img_tk)
        lbl.image = img_tk
        lbl.grid(row=i//2, column=i%2, padx=5, pady=5)

def save_images():
    if not generated_images:
        messagebox.showwarning("Warning", "Generate images first!")
        return
    
    folder_path = filedialog.askdirectory()
    if folder_path:
        for i, img in enumerate(generated_images):
            img = Image.fromarray(img.squeeze())
            img_path = os.path.join(folder_path, f"generated_image_{i+1}.png")
            img.save(img_path)
        messagebox.showinfo("Saved", "Images saved successfully!")

# Create Tkinter window
root = tk.Tk()
root.title("GAN Image Generator")
root.geometry("400x500")
root.resizable(False, False)

status_label = tk.Label(root, text="Load a GAN model to begin", fg="blue")
status_label.pack(pady=10)

load_button = tk.Button(root, text="Load Model", command=load_model, bg="lightgray")
load_button.pack(pady=5)

generate_button = tk.Button(root, text="Generate Images", command=generate_images, bg="lightblue")
generate_button.pack(pady=5)

image_frame = tk.Frame(root)
image_frame.pack(pady=10)

save_button = tk.Button(root, text="Save Images", command=save_images, bg="lightgreen")
save_button.pack(pady=5)

root.mainloop()
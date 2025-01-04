import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Function to preprocess the image for prediction
def preprocess_image(img_path):
    img = Image.open(img_path)
    img = img.resize((150, 150))  # Resize image to match model's input size
    img_array = np.array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Function to make predictions
def make_prediction():
    try:
        # Preprocess the image
        img_array = preprocess_image(uploaded_image_path)

        # Make predictions
        prediction = model.predict(img_array)
        prediction = (prediction > 0.5).astype(int)  # Convert prediction to binary class (0 or 1)

        # Display the result in a message box
        if prediction == 1:
            result = "Dog"
        else:
            result = "Cat"

        messagebox.showinfo("Prediction", f"The image is a {result}.")

    except Exception as e:
        messagebox.showerror("Error", f"Error making prediction: {e}")

# Function to upload the image and display it
def upload_image():
    global uploaded_image_path
    # Ask the user to upload an image file
    uploaded_image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not uploaded_image_path:
        return

    try:
        # Open and display the uploaded image
        img = Image.open(uploaded_image_path)
        img.thumbnail((250, 250))  # Resize to fit the window
        img = ImageTk.PhotoImage(img)

        # Update the label with the image
        image_label.config(image=img)
        image_label.image = img  # Keep a reference to the image to prevent it from being garbage collected

    except Exception as e:
        messagebox.showerror("Error", f"Error uploading image: {e}")

# Set up the Tkinter window
window = tk.Tk()
window.title("Cat or Dog Classifier")

# Set the window size
window.geometry("400x400")

# Create a button to upload an image
upload_button = tk.Button(window, text="Upload Image", command=upload_image)
upload_button.pack(pady=10)

# Create a button to make the prediction
predict_button = tk.Button(window, text="Predict", command=make_prediction)
predict_button.pack(pady=10)

# Label to show the uploaded image
image_label = tk.Label(window)
image_label.pack(pady=10)

# Run the Tkinter event loop
window.mainloop()

import tkinter as tk
from tkinter import scrolledtext
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import pickle

# Load the trained model
with open('chatbot_model_improved.pkl', 'rb') as file:
    model = pickle.load(file)

def get_response(user_input):
    response = model.predict([user_input])[0]
    return response

def submit_input():
    user_input = entry.get()
    if user_input:
        chat_window.config(state=tk.NORMAL)
        chat_window.insert(tk.END, "You: " + user_input + "\n", 'user')
        chat_window.config(state=tk.DISABLED)
        entry.delete(0, tk.END)
        program_response = get_response(user_input)
        chat_window.config(state=tk.NORMAL)
        chat_window.insert(tk.END, "Bot: " + program_response + "\n", 'bot')
        chat_window.config(state=tk.DISABLED)
        chat_window.yview(tk.END)

# Set up the GUI
root = ttk.Window(themename="litera")
root.title("Mental Health Chatbot")
root.geometry("500x600")

# Define colors
bg_color = "#f0f4f8"  # Light grey background
text_color = "#333333"  # Dark grey text
entry_bg_color = "#ffffff"  # White entry background
button_color = "#4CAF50"  # Green button
button_hover_color = "#45a049"  # Darker green button on hover
title_color = "#2196F3"  # Blue title
user_text_color = "#3F51B5"  # Indigo user text
bot_text_color = "#009688"  # Teal bot text

# Configure the root window background color
root.configure(bg=bg_color)

# Title Label
title_label = ttk.Label(root, text="Mental Health Chatbot", font=("Helvetica Neue", 20, "bold"), background=bg_color, foreground=title_color)
title_label.pack(pady=20)

# Create a scrolled text widget for the chat window
chat_window = scrolledtext.ScrolledText(root, state=tk.DISABLED, wrap=tk.WORD, width=60, height=20, bg=entry_bg_color, fg=text_color, font=("Helvetica Neue", 12), relief=tk.FLAT)
chat_window.pack(padx=10, pady=10)

# Configure tags for text color
chat_window.tag_config('user', foreground=user_text_color, font=("Helvetica Neue", 12, "bold"))
chat_window.tag_config('bot', foreground=bot_text_color, font=("Helvetica Neue", 12))

# Create an entry widget for user input
entry_frame = ttk.Frame(root)
entry_frame.pack(padx=10, pady=10, fill=tk.X)

entry = ttk.Entry(entry_frame, width=40, font=("Helvetica Neue", 14), background=entry_bg_color, foreground=text_color)
entry.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.X, expand=True)
entry.bind("<Return>", lambda event: submit_input())

# Function to change button color on hover
def on_enter(e):
    submit_button['bootstyle'] = 'success'

def on_leave(e):
    submit_button['bootstyle'] = 'outline-success'

# Create a button to submit the input
submit_button = ttk.Button(entry_frame, text="Send", command=submit_input, bootstyle="outline-success", width=8)
submit_button.pack(side=tk.RIGHT, padx=10, pady=10)

# Bind hover effect to the button
submit_button.bind("<Enter>", on_enter)
submit_button.bind("<Leave>", on_leave)

# Start the main event loop
root.mainloop()

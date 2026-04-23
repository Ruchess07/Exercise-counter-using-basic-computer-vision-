# Exercise-counter-using-basic-computer-vision-
Real-time exercise rep counter using computer vision and OpenCV. Tracks user-selected body movement via webcam and counts repetitions automatically, eliminating manual counting and helping users focus on form and mind-muscle connection during workouts.

This is a real time exercise rep counter built with Python, OpenCV, and a webcam. It tracks a user selected body part or object, detects repeated motion, and counts reps automatically so users can focus on form and mind muscle connection instead of counting manually.

## Features

- Real time webcam based tracking
- Manual region selection for flexible exercise setup
- Automatic rep counting using motion cycles
- Smoothing to reduce noisy movement signals
- Simple keyboard controls for reset and reselection
- Lightweight setup with no external model files

## Tech Stack

- Python
- OpenCV
- NumPy

## How It Works

The app opens a webcam feed and allows the user to draw a box around the moving body part, such as a hand, forearm, dumbbell, or torso. It then tracks that region across frames and measures movement over time. When the tracked motion completes a full cycle, the app increments the rep count.


Nuclei Segmentation with U-Net

This is a learning project. I had not worked on image segmentation before, so I built a small working model to understand it instead of only reading about it. The code is kept simple on purpose, and the notes below are what I actually learned.

The task

Three different jobs, easy to mix up:

Classification says what is in a picture.
Detection draws a box around it.
Segmentation labels every single pixel.

So this model does not give back a number. It gives back a picture, the same size as the one you put in, white where the nuclei are and black everywhere else. That picture is called a mask.

The data

Data Science Bowl 2018 — microscope photos of cells. 664 photos, 28,986 nuclei in total, so about 44 in each photo.

I picked this one because the nuclei are small, there are many of them, and they touch each other. Finding them is the easy half. Telling two touching ones apart is the hard half, and that is the interesting part.

The 664 images are split into 531 for training and 133 for validation (80/20, random seed 42).

The model

A U-Net with a ResNet-34 encoder, using segmentation-models-pytorch.

A U-Net works in two halves. Going down, it shrinks the picture step by step until it understands what it is looking at. Going back up, it grows it to full size again to say where. Down and up — that shape is where the name comes from.

The clever bit is the skip connections. Once the picture has been shrunk right down, the model knows what is in it but has lost the sharp edges. So the detail from the way down is passed straight across to the way up, and the edges come back.

Scoring

Accuracy is the wrong measure here, and it is worth seeing why.

Nuclei fill only a small part of each photo. So a lazy model that answers "background" for every single pixel still gets most pixels right — a great score for something completely useless.

Instead I use IoU: how much the predicted mask and the true mask overlap, divided by the area they cover together. Guess nothing and you score zero. Dice is the same idea, scored a little more kindly.

Notes as I go

Look at the data before writing any code for it. The copy I could download had been repacked by someone else, and it was nothing like the original — files dumped into one flat folder, the whole thing buried two folders deep, and the labels saved as JSON instead of mask pictures. Everything I expected from the tutorials was wrong. Half an hour of just listing folders saved hours of confusion later.

The labels come one nucleus at a time. There is no ready-made mask. There are 28,986 separate outlines, each one saying "this shape, in this photo, is one nucleus". To train the model, all the outlines in a photo get merged into one black-and-white mask.

Merge with maximum, not plus. Each nucleus gets painted onto a blank sheet, and at every pixel I keep the bigger of the two values. Since the values are only 0 or 1, that means "white if either is white". Adding instead would give a 2 wherever two outlines overlap, and the mask would no longer be clean black and white.

That merge also throws away the interesting part. The original labels know which pixels belong to which nucleus. The merged mask only knows nucleus or not. Getting from "these pixels are nuclei" back to "these pixels are this one nucleus" is the whole reason touching objects are hard.

How to run

Made for Google Colab with a GPU (Runtime → Change runtime type → T4 GPU).

The dataset zip is stored in Google Drive. When you open the notebook, mount Drive when Colab asks, then run all cells top to bottom. No Kaggle token needed — the data unzips from Drive automatically each session.
